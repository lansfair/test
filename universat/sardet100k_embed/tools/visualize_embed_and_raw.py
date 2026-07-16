from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image


def _normalize_for_display(
    arr: np.ndarray,
    percentiles: tuple[float, float] = (1, 99),
) -> np.ndarray:
    out = np.empty_like(arr, dtype=np.float32)
    for c in range(arr.shape[-1]):
        channel = arr[..., c]
        vmin = float(np.nanpercentile(channel, percentiles[0]))
        vmax = float(np.nanpercentile(channel, percentiles[1]))
        if vmax - vmin < 1e-8:
            out[..., c] = 0.0
        else:
            out[..., c] = np.clip(
                (channel - vmin) / (vmax - vmin), 0.0, 1.0,
            )
    return out


def _resize_to_target(
    arr: np.ndarray,
    target_h: int,
    target_w: int,
) -> np.ndarray:
    channels = arr.shape[0]
    resized = np.empty((channels, target_h, target_w), dtype=arr.dtype)
    for c in range(channels):
        resized[c] = np.array(
            Image.fromarray(arr[c]).resize(
                (target_w, target_h),
                Image.BILINEAR,
            )
        )
    return resized


def _read_first3(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        band_count = int(src.count)
        bands = list(range(1, min(band_count, 3) + 1))
        return src.read(bands)


def _make_overlay(
    raw_rgb: np.ndarray,
    embed_rgb: np.ndarray,
) -> np.ndarray:
    raw_h, raw_w = raw_rgb.shape[:2]
    e_h, e_w = embed_rgb.shape[:2]
    overlay = raw_rgb.copy().astype(np.float32) * 0.4
    for c in range(3):
        overlay[:, :, c] += embed_rgb[:, :, c] * 0.6
    return np.clip(overlay, 0.0, 1.0)


def visualize_sample(
    embed_path: Path,
    raw_path: Path,
    output_path: Path,
    sample_name: str,
) -> None:
    embed_chw = _read_first3(embed_path)
    raw_chw = _read_first3(raw_path)

    _, raw_h, raw_w = raw_chw.shape

    # Resize embedding to match raw spatial resolution (patch_size=4 → 200→800)
    embed_up_chw = _resize_to_target(embed_chw, raw_h, raw_w)

    # Convert CHW → HWC and normalize
    raw_hwc = raw_chw.transpose(1, 2, 0).astype(np.float32) / 255.0
    raw_hwc = np.clip(raw_hwc, 0.0, 1.0)

    embed_hwc = embed_up_chw.transpose(1, 2, 0)
    embed_hwc = _normalize_for_display(embed_hwc)

    overlay_hwc = _make_overlay(raw_hwc, embed_hwc)

    fig = plt.figure(figsize=(16, 5))
    gs = GridSpec(1, 3, figure=fig, wspace=0.04)

    titles = ["Raw Input (RGB, 800×800)", "OLMoEarth Embedding (ch 0–2, upsampled)", "Overlay"]
    images = [raw_hwc, embed_hwc, overlay_hwc]

    for idx, (ax_title, img) in enumerate(zip(titles, images)):
        ax = fig.add_subplot(gs[0, idx])
        ax.imshow(img)
        ax.set_title(ax_title, fontsize=10)
        ax.axis("off")

    fig.suptitle(sample_name, fontsize=12, y=0.98)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize embedding (first 3ch) and raw GeoTIFF RGB side by side per sample.",
    )
    parser.add_argument(
        "--input-root",
        required=True,
        help=(
            "Directory containing split subdirectories (train/val/test), "
            "each with numbered sample folders."
        ),
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Output directory for PNG visualizations.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train"],
        choices=["train", "val", "test"],
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Process at most N samples per split (0 = all).",
    )
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)

    for split in args.splits:
        split_dir = input_root / split
        if not split_dir.is_dir():
            print(f"[skip] split directory not found: {split_dir}")
            continue

        sample_dirs = sorted(
            [d for d in split_dir.iterdir() if d.is_dir()],
            key=lambda p: p.name,
        )
        total = len(sample_dirs)
        if args.max_samples > 0:
            sample_dirs = sample_dirs[: args.max_samples]

        done = 0
        skipped = 0
        for sample_dir in sample_dirs:
            embed_path = sample_dir / "embedding.tif"
            raw_path = sample_dir / "raw_input.tif"
            if not embed_path.exists() or not raw_path.exists():
                skipped += 1
                continue

            output_path = output_root / split / f"{sample_dir.name}.png"
            visualize_sample(
                embed_path,
                raw_path,
                output_path,
                f"{split}/{sample_dir.name}",
            )
            done += 1
            if done % 200 == 0:
                print(f"[{split}] {done}/{len(sample_dirs)} done")

        print(
            f"[{split}] finished: {done} visualized, {skipped} skipped "
            f"(total {len(sample_dirs)} samples)"
        )

    print("done.")


if __name__ == "__main__":
    main()
