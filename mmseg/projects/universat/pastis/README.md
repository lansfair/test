# UniverSat on PASTIS-R

This project evaluates UniverSat on the PASTIS-R dataset for semantic
segmentation. It follows the same structure as `projects/copernicus/pastis`
and `projects/olmoearth/pastis`.

## Structure

```
pastis/
├── universat_pastis/           # Python package for PASTIS-R components
│   ├── datasets/pastis.py      # UniverSatPASTISDataset + collate function
│   ├── transforms/formatting.py
│   └── utils/norm.py
├── configs/
│   ├── universat-base_pastis_lp.py  # Linear probe (frozen backbone)
│   └── universat-base_pastis_ft.py  # Fine-tune
├── train.sh
└── test.sh
```

The backbone, decode heads, and data preprocessor are reused from
`projects/universat/universat`.

## Data layout

Your PASTIS-R directory should look like this::

```
PASTIS-R/
  metadata.geojson              # ID_PATCH, Fold, dates-S2, dates-S1A
  DATA_S2/S2_{id}.npy           # T x 10 x H x W
  DATA_S1A/S1A_{id}.npy         # T x 3 x H x W
  ANNOTATIONS/TARGET_{id}.npy   # 1 x H x W or H x W
  NORM_S2_patch.json            # {"mean": [...], "std": [...]}
  NORM_S1_patch.json            # {"mean": [...], "std": [...]}
```

If you do not have normalization JSON files, compute them from the training
split first or temporarily set `norm_path=None` in the configs (not recommended
for real evaluation).

## Usage

Set the environment variables in `train.sh` / `test.sh` (especially
`MM_ARCHIVE_DATA_HOME`, `MM_ARCHIVE_CKPT_HOME`, `CONDA_ENV_NAME`, and
`CUDA_VISIBLE_DEVICES`) and run:

```bash
cd projects/universat/pastis
bash train.sh
```

Or manually from the MMSegmentation root::

```bash
export PYTHONPATH=".:$PWD/projects/universat:$PWD/projects/universat/pastis:$PYTHONPATH"
python tools/train.py \
    projects/universat/pastis/configs/universat-base_pastis_lp.py \
    --work-dir work_dirs/universat-base_pastis_lp
```

## Configs

- `universat-base_pastis_lp.py`: backbone frozen (`frozen_stages=0`), only the
  linear probe head is trained. Use this for standard LP evaluation.
- `universat-base_pastis_ft.py`: backbone unfrozen (`frozen_stages=-1`), the
  whole model is fine-tuned with a small conv segmentation head.

To switch to UniverSat-Tiny, change `embed_dim` to 192, `num_heads` to 8, and
`block_type` to `("Bi_ACA_in", "SAx12", "Bilinear_out", "CA_Sub")` (the default
Base block already has 12 SA blocks; Tiny has 6).

## Notes

- PASTIS-R samples have variable time-series length. A custom collate function
  (`universat_pastis_collate`) pads each modality and its dates to the max
  length in the batch.
- The input dict passed to the backbone contains both modality tensors
  (`s2`, `s1`) and their date tensors (`s2_dates`, `s1_dates`).
- `output_grid=128` means the backbone outputs a 128 x 128 token grid,
  matching the native PASTIS-R 128 x 128 resolution.
- `num_classes=20` and `ignore_index=19`: PASTIS-R has 20 labels (0=background,
  1-18=crops, 19=void). The void class is ignored; background is treated as a
  valid class. Adjust if your annotation convention differs.
