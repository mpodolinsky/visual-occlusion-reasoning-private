# CLIP text-alignment for the perception probe

Goal: give the probe's latent `z` a CLIP-text-space geometry so it can be
**text-queried for steering** (retrieve nearest language description, inject into
the VLA prompt), following the paper's `L_align` (VLM-AD style). Detection alone
only yields a scalar; alignment makes `z` semantically addressable.

Status: **mechanism solved, supervision pending.** The v2 recipe (see bottom)
makes `z` a near-perfect CLIP-text-queryable latent for seen tasks (100% top-1
retrieval) with **zero** detection cost. But only the 10 task instructions are
available as anchors, so `z` currently encodes "which task," not "which failure
mode." The steering payoff needs failure-mode text (taxonomy / per-timestep VLM
captions, not yet collected) — at which point the v2 recipe should transfer.

---

## Architecture ("Option A": `z` in the shared trunk, 512-d, direct alignment)

`--embed-dim 512` splits the MLP head into a named bottleneck:

```
pooled(6144) → LayerNorm → [embed] → z(512) → GELU → Dropout → [classifier] → 1
```

- `--embed-dim D`         : size of `z` (512 to match CLIP ViT-B/32 text dim).
- `--embed-hidden H`      : compression layer `6144→H→D` inside `[embed]`. Without
  it the bare `6144→512` doubles head params vs the default 256 head.
- `forward(..., return_embedding=True)` returns `(logits, z)`.
- Default construction (no `--embed-dim`) is byte-identical to the original probe.

Code: `probe_model.py` (`embed_dim`/`embed_hidden`), `train_probe_time_dependent.py`
(`episode_embedding`, `clip_alignment_loss`, `--align-weight`, `--align-temp`,
`--clip-embeddings`).

## Alignment loss

Per training episode:
1. mean-pool `z` over valid timesteps → one 512-d vector (`episode_embedding`,
   subsamples ≤12 evenly-spaced steps to avoid a 2nd full B·T forward → OOM).
2. CLIP ViT-B/32 text-encode the 10 task instructions → 10 fixed L2-norm anchors.
3. InfoNCE: `CE(normalize(z) @ anchors.T / 0.07, task_idx)`.

`total = raw_target_BCE + align_weight · InfoNCE`. Gradient from InfoNCE reaches
`[embed]` only, not `[classifier]`.

Anchors: `scripts/perception_probe/clip_task_embeddings.py` →
`outputs/perception_probe/clip/` (run with `submodules/openpi/.venv`).

---

## CLIP-space sanity check (cached anchors, seed-0 split)

Nearest **seen** task for each **unseen** task (cosine):

| unseen task | nearest seen | cos |
|---|---|---|
| alphabet soup + tomato sauce | alphabet soup + cream cheese | 0.943 |
| cream cheese + butter (13 of 16 unseen failures) | alphabet soup + cream cheese | 0.928 |
| pick up book / caddy | alphabet soup + cream cheese | 0.779 |

→ 2/3 unseen tasks (incl. the failure-heavy one) sit close to a seen task, so
alignment **can** transfer. The book/caddy task is a CLIP-space island.

Success vs failure phrasing: `cos("successfully X", "failed to X") = 0.986` vs
`cos(success_i, success_j) = 0.742`. **CLIP is nearly blind to the outcome verb**
— a task×outcome anchor scheme will not teach failure semantics. Need real
failure-mode descriptions.

---

## Results

Metric = zero-shot **unseen**-split separation `(mean_fail − mean_success)/std`
at the task-min-step cutoff, plus unseen AUROC. seed 0, batch 8, ~10 epochs,
best-val-AUROC checkpoint. **cap-80** = episodes clipped to 80 steps (a training
handicap); **uncapped** is the real reference.

### Architecture gate — does the 512 layer degrade detection? (cap-80)

| variant | test AUROC | unseen sep | unseen AUROC |
|---|---|---|---|
| default (no `z` layer) | 0.839 | 1.59 | 0.904 |
| naive `6144→512` (bare) | 0.791 | **0.05** | **0.549** |
| naive `6144→512`, dropout 0.2 | 0.821 | 0.21 | 0.502 |
| naive, 5 epochs | 0.725 | 0.56 | 0.611 |
| naive, `--select-by separation` | 0.755 | 0.27 | 0.576 |
| **bottleneck `6144→256→512→256→1`** | 0.923 | 1.31 | 0.864 |
| bottleneck `6144→128→512→…` | 0.908 | 1.26 | 0.874 |
| bottleneck, 6 epochs | 0.859 | 0.96 | 0.776 |

- **Naive 512 layer breaks zero-shot detection** (~2M extra head params memorise
  the ~210 train episodes; unseen AUROC = coin flip). No epoch/dropout/checkpoint
  trick recovers it.
- **Bottleneck-first recovers** to ~3–4 pts below default. `embed_hidden=256`
  ≈ `128`. Fewer epochs hurt (needs the full ~8).

### Alignment weight sweep — `embed512_bn256` + `L_align` (cap-80)

| align_weight | test AUROC | unseen sep | unseen AUROC | best ep |
|---|---|---|---|---|
| 0 (bottleneck only) | 0.923 | 1.31 | 0.864 | 8 |
| **0.1** | 0.905 | **1.78** | **0.911** | 8 |
| 0.3 | 0.912 | 1.65 | 0.893 | 6 |
| 1.0 | 0.866 | 1.50 | 0.859 | 6 |

- **align 0.1 closes the gate's detection gap and beats the cap-80 default**
  (1.78 vs 1.59). The CLIP anchors act as a regulariser on `z`, countering the
  bottleneck's extra capacity.
- Monotone: **more alignment pressure -> less detection benefit** (1.78 -> 1.65
  -> 1.50). 0.1 is the sweet spot; even 1.0 stays above bottleneck-only.

### Uncapped apples-to-apples

Batch-8 runs OOM at eval when the pi0.5 server holds the card (~160 MB free);
these are **batch 6** (costs ~0.3 sep vs batch 8 — see the reference row).
`score_sequence` now chunks the eval forward + casts bf16 per chunk (bit-identical).

| run (uncapped) | test AUROC | unseen sep | unseen AUROC | ep |
|---|---|---|---|---|
| default, batch 8 (phase 5/6 reference) | 0.907 | 2.14 | 0.926 | 7 |
| default, batch 6 | 0.852 | 1.81 | 0.893 | 6 |
| **default + `--n-queries 4` only**, batch 6 | **0.941** | **1.94** | **0.923** | 7 |
| `embed512_bn256` + align 0.1, batch 6 | 0.879 | 1.79 | 0.867 | 6 |

### Batch-8 uncapped (2.14 regime)

| run (batch 8, uncapped) | test AUROC | unseen sep | unseen AUROC | ep |
|---|---|---|---|---|
| default (control) | 0.907 | **2.14** | 0.926 | 7 |
| default + `--n-queries 4` | 0.903 | 1.90 | 0.916 | 5 |
| `--n-queries 4` + align 0.1 + bn256 (batch 6) | 0.903 | **0.21** | **0.535** | 7 |

- The batch-8 `default` control **exactly reproduces 2.14 / 0.926** → the
  eval-chunking + `empty_cache` changes are confirmed bit-identical, and the
  batch 8 → 6 drop (2.14 → 1.81) is a real effect, not a code artifact.
- **`--n-queries 4` is NOT a robust win.** batch 6: 1.94 > 1.81 default. batch 8:
  1.90 < 2.14 default. The sign flips with batch size → it's inside
  checkpoint/batch noise, not a real improvement. Earlier "combo" still clearly
  hurts (share_image_pool + adamw).
- **align 0.1 stays detection-neutral** across both cap-80 (1.78) and batch-6
  uncapped (1.79); a batch-8 uncapped align run is still pending (OOM).
- **`--n-queries 4` + align 0.1 do NOT compose** — stacked, unseen collapses to
  0.21 / 0.535 (coin flip) while val + test stay healthy (0.90). The two
  capacity adds (4 pooling heads + the 512-d bottleneck) together overfit to
  task identity. Each alone is fine; together is a trap. Also: val AUROC climbs
  to 0.907 through this collapse → **val is blind to it**, so checkpoint
  selection can't save you here.

### Is `z` actually text-queryable? (`eval_clip_retrieval.py`)

10-way task retrieval: `argmax_i cos(mean-pooled z, CLIP_anchor_i)` vs the
episode's own task. Chance = 0.10. (train-time `align_acc` is wandb-only; this
recovers it post-hoc from `probe_best.pt` + `split.json`.)

| model | train top1 | test top1 | test top3 | unseen top1 |
|---|---|---|---|---|
| bn256, no align | 0.10 | 0.09 | 0.25 | 0.03 |
| align 0.1 (cap-80) | 0.20 | 0.21 | 0.51 | **0.00** |
| align 0.3 (cap-80) | 0.36 | 0.32 | 0.72 | **0.00** |
| align 1.0 (cap-80) | 0.41 | 0.42 | 0.74 | **0.00** |
| align 0.1 (uncapped b6) | 0.15 | 0.11 | 0.49 | **0.00** |

- **Retrieval is weak even on seen tasks** — best case (weight 1.0) is 42%
  top-1, and that's the weight that most hurts detection. `cos(z, correct
  anchor)` stays near 0 (+0.01 to +0.07) — InfoNCE only shapes *relative*
  similarity at temp 0.07, so absolute geometry is loose and nearest-anchor
  retrieval is noisy.
- **Unseen retrieval is exactly 0.00 for every align weight.** The 3 unseen
  anchors are in the candidate set, but unseen `z` never points at them
  (`cos(z, true)` ≈ 0 or negative). Alignment learned a *seen-task-specific*
  z→text map that does not transfer at all.
- Retrieval accuracy trades directly against detection: 0.1 → 1.0 lifts test
  top1 0.21 → 0.42 but drops sep 1.78 → 1.50.

**Conclusion (v1): this InfoNCE-only + raw-anchor setup does not produce a
usefully text-queryable `z`.** The 10 raw CLIP anchors are near-collinear
(off-diag cos 0.78) so nearest-anchor retrieval tops out at 42% on seen tasks
and costs detection. **v2 (below) fixes this** — mean-centering the anchors +
a direct cos term + a properly decoupled z-head gets seen-task retrieval to
~100% at zero detection cost. Unseen retrieval stays ~0 regardless (expected:
task-only anchors, no failure semantics — still needs real failure-mode text
for the steering payoff).

---

## v2 (`clip_align_v2/`): centering + cos-term + decoupling  ← SOLVED

Fresh copy, nothing in the main pipeline touched. Four flags:
1. `--align-center-anchors` — mean-center the CLIP anchors (off-diag cos 0.78 -> -0.11).
2. `--align-cos-weight W` — add `W*(1 - cos(z, c_true))` so z lands ON its anchor.
3. `--decouple-z` — z is a parallel head off `pooled`; detection head = the
   original single-stage head.
4. `--z-stop-grad` — feed `pooled.detach()` into the z head. Align loss trains
   ONLY z_head; pools + detection head get zero align gradient.
   `--reseed-after-build` — `torch.manual_seed(seed)` after `build_model`, so
   the decoupled model (which draws ~1.7M extra RNG for z_head init) trains
   against the SAME dropout/shuffle stream as a plain probe. Without it the
   decoupled runs are a fixed unlucky draw (see below).

All uncapped, batch 6, seed 0. Retrieval = 10-way task, chance 0.10, anchors
mean-centered.

### The winning recipe

```
--decouple-z --z-stop-grad --reseed-after-build --align-center-anchors
--align-weight 3 --align-cos-weight 2
```

| run | align wt | detection sep / AUROC | seen-test retr. top1 / top3 | cos(z, anchor) |
|---|---|---|---|---|
| plain default (b6, reseeded) | — | 1.90 / 0.931 | — | — |
| decouple+stopgrad, align **off** (reseeded) | 0 | **1.90 / 0.931** (byte-identical) | — | — |
| `final_w1` | 1.0 | 1.88 / 0.938 | 0.925 / 0.981 | +0.70 |
| **`final_w3`** | 3.0 | **1.88 / 0.938** | **0.981 / 1.000** | +0.79 |
| `final_w8` | 8.0 | 1.88 / 0.938 | 1.000 / 1.000 | +0.86 |

- **Both objectives at once.** Detection 1.88 / 0.938 — *above* the plain default
  (1.81) — and byte-identical across align weights 1/3/8 (stop-grad makes the
  weight touch only z_head). Seen-task retrieval up to 100% top-1 with z on its
  anchor (cos 0.86). Old best was 42%, and that cost detection.
- Weight 3 is the knee (98% / 100%); weight 8 buys the last 2% for nothing.
- Unseen retrieval still 0% top-1 (top-3 crept to ~0.2) — expected, not a goal.

### How we got here (and the decoupling red herring)

| run | config | unseen sep | note |
|---|---|---|---|
| old `align_0.1` | in-trunk, raw anchors | 1.78 | retrieval capped 21% |
| `intrunk_w0.1` | in-trunk, center+cos, wt 0.1 | 1.86 | detection kept, retrieval only 28% |
| `all_w1` / `all_w0.3` | center+cos, decouple, wt 1.0 / 0.3 | 0.80 / 0.68 | retrieval ~100%, detection looked broken |
| `sg_w1` / `sg_w3` | + `--z-stop-grad`, wt 1.0 / 3.0 | 0.64 / 0.64 | identical -> weight has 0 effect on detection |
| `ctrl_decouple_noalign` | decouple+stopgrad, align OFF | 0.76 | still low with NO alignment at all |

The `all_*` runs really do hurt detection (align gradient reshapes the shared
pools). But `sg_*` and the align-OFF control were the puzzle — stop-grad gives
provably zero align gradient into pools/head, yet detection sat at 0.64-0.76.

**Cause: an RNG-stream shift, not the architecture.** Verified two ways:
- Init weights of the detection subnet (pools + head) are **byte-identical**
  between plain and decoupled at the same seed (max|Δ| = 0). But `torch.rand()`
  right after `build_model` differs — building z_head draws ~1.7M numbers and
  advances the global generator, so every downstream dropout mask / shuffle
  comes from a shifted stream.
- With `--reseed-after-build` on both, `plain` and `decouple+stopgrad(align 0)`
  train to **sep 1.90 / AUROC 0.931 / test 0.872 / ep 7 — identical to 3
  decimals.**

So decoupling is free; the 5 low decoupled numbers were one deterministic
unlucky dropout sequence. Corollary: **this regime's run-to-run noise is ~±1.0
separation** — a pure reshuffle at the same seed moved unseen sep 1.90 -> 0.76.
Every single-seed number in this doc has that error bar.

---

## Open items

- **Real failure-mode / per-timestep VLM-caption anchors.** The v2 recipe proves
  the mechanism works (100% retrieval on the anchors it's trained on, detection
  untouched). But task-instruction anchors only make `z` a task-ID vector; the
  steering payoff needs `z` to retrieve *failure descriptions*, which needs that
  text collected.
- **Multi-seed the v2 recipe.** Everything is n=1 in a ±1.0-separation regime.
  `final_w3` (sep 1.88) should be run at ≥3 seeds before it goes in the paper.
- **Port v2 into the main trainer** once the anchor text exists — currently
  `clip_align_v2/` is a copy. The four flags (`--decouple-z --z-stop-grad
  --reseed-after-build --align-center-anchors`) + `--align-cos-weight` are the
  whole diff.
- `--reseed-after-build` should probably be default-on everywhere — it's the
  honest thing (arch changes shouldn't silently shift the RNG stream) and it
  made plain-vs-decoupled byte-identical.
