#!/usr/bin/env python3
"""Build one 2x5 image grid for each rendered LIBERO-Occ suite."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "libero_occ_tasks",
        help="Directory containing one subdirectory per suite.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "libero_occ_composites",
        help="Directory in which suite composite PNGs are written.",
    )
    return parser.parse_args()


def task_number(path: Path) -> int:
    try:
        return int(path.name.split("-", 1)[0])
    except ValueError as error:
        raise ValueError(f"Image lacks a numeric task prefix: {path}") from error


def make_composite(image_paths: list[Path], output_path: Path) -> None:
    if len(image_paths) != 10:
        raise ValueError(
            f"Expected 10 task images for {output_path.stem}, found {len(image_paths)}"
        )

    images: list[Image.Image] = []
    try:
        for path in image_paths:
            images.append(Image.open(path).convert("RGB"))

        tile_size = images[0].size
        if any(image.size != tile_size for image in images):
            raise ValueError(f"Task images have inconsistent dimensions for {output_path.stem}")

        tile_width, tile_height = tile_size
        composite = Image.new("RGB", (tile_width * 2, tile_height * 5))
        for index, image in enumerate(images):
            column = index % 2
            row = index // 2
            composite.paste(image, (column * tile_width, row * tile_height))

        output_path.parent.mkdir(parents=True, exist_ok=True)
        composite.save(output_path)
    finally:
        for image in images:
            image.close()


def main() -> int:
    args = parse_args()
    suites = sorted(path for path in args.input_dir.resolve().iterdir() if path.is_dir())
    if not suites:
        raise FileNotFoundError(f"No suite directories found in {args.input_dir}")

    output_dir = args.output_dir.resolve()
    for suite in suites:
        image_paths = sorted(suite.glob("*.png"), key=task_number)
        output_path = output_dir / f"{suite.name}.png"
        make_composite(image_paths, output_path)
        print(f"Saved {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
