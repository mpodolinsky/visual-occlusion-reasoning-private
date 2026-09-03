#!/usr/bin/env python3
"""Compute CLIP ViT-B/32 text embeddings for the libero_10_occluded task
instructions (and task x outcome variants), cache them, and sanity-check the
geometry before wiring an alignment loss:

  1. Do the 10 task instructions form sensible clusters?
  2. Do "successfully X" / "failed to X" separate for the same task?
  3. For each seed-0 UNSEEN task, what's its nearest SEEN task in CLIP space?
     (if unseen tasks are islands, alignment can't help them transfer.)

Run with the openpi submodule venv (has transformers):
    submodules/openpi/.venv/bin/python scripts/perception_probe/clip_task_embeddings.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs" / "perception_probe" / "clip"
MANIFEST = REPO / "outputs" / "perception_probe" / "features" / "manifest.csv"

# seed-0 unseen tasks (from any sweep split.json)
UNSEEN = {
    "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy",
    "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket",
    "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket",
}


def task_to_instruction(task: str) -> str:
    """KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it
       -> 'turn on the stove and put the moka pot on it'"""
    parts = task.split("_")
    # drop leading SCENE-name tokens: everything up to and incl. the 'SCENEn' token
    for i, p in enumerate(parts):
        if p.upper().startswith("SCENE"):
            return " ".join(parts[i + 1 :]).lower()
    return " ".join(parts).lower()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    from transformers import CLIPModel, CLIPTokenizer

    tasks = sorted(
        {r["task"] for r in csv.DictReader(open(MANIFEST)) if r["suite"] == "libero_10_occluded"}
    )
    instr = {t: task_to_instruction(t) for t in tasks}
    print(f"{len(tasks)} tasks:")
    for t in tasks:
        tag = "UNSEEN" if t in UNSEEN else "seen  "
        print(f"  [{tag}] {instr[t]}")

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
    tok = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    def embed(texts: list[str]) -> np.ndarray:
        with torch.no_grad():
            batch = tok(texts, padding=True, truncation=True, return_tensors="pt")
            z = model.get_text_features(**batch)
        z = torch.nn.functional.normalize(z, dim=-1)
        return z.cpu().numpy()

    # --- 1. plain task instructions ---
    E = embed([instr[t] for t in tasks])  # (10, 512), L2-normalised
    np.save(OUT / "task_instruction_embeddings.npy", E)
    (OUT / "task_index.json").write_text(
        json.dumps({t: i for i, t in enumerate(tasks)}, indent=2)
    )

    sim = E @ E.T
    print("\n=== task x task cosine similarity ===")
    hdr = "         " + " ".join(f"{i:>5d}" for i in range(len(tasks)))
    print(hdr)
    for i, t in enumerate(tasks):
        row = " ".join(f"{sim[i, j]:5.2f}" for j in range(len(tasks)))
        print(f"{i:2d} {'U' if t in UNSEEN else ' '} {row}   {instr[t][:40]}")

    # --- 3. nearest seen task for each unseen task ---
    seen_idx = [i for i, t in enumerate(tasks) if t not in UNSEEN]
    print("\n=== nearest SEEN task for each UNSEEN task ===")
    for i, t in enumerate(tasks):
        if t not in UNSEEN:
            continue
        sims = [(sim[i, j], tasks[j]) for j in seen_idx]
        sims.sort(reverse=True)
        print(f"  UNSEEN: {instr[t]}")
        for s, st in sims[:3]:
            print(f"     {s:.3f}  {instr[st]}")

    # --- 2. task x outcome ---
    succ = [f"the robot successfully {instr[t]}" for t in tasks]
    fail = [f"the robot failed to {instr[t]}" for t in tasks]
    Es, Ef = embed(succ), embed(fail)
    np.save(OUT / "task_outcome_success_embeddings.npy", Es)
    np.save(OUT / "task_outcome_fail_embeddings.npy", Ef)
    diag = (Es * Ef).sum(-1)  # cos(success_i, fail_i) per task
    cross_s = (Es @ Es.T)
    cross_f = (Ef @ Ef.T)
    print("\n=== success vs failure description separation ===")
    print(f"  mean cos(success_i, fail_i)  [same task, opp outcome] : {diag.mean():.3f}")
    off_s = cross_s[~np.eye(len(tasks), dtype=bool)].mean()
    print(f"  mean cos(success_i, success_j) [diff task, same outcome]: {off_s:.3f}")
    print("  -> if the first is HIGHER, CLIP barely distinguishes success/failure phrasing")

    print(f"\nsaved embeddings + index to {OUT}/")


if __name__ == "__main__":
    main()
