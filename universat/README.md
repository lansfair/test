# UniverSat for MMSegmentation 1.x

This project migrates the UniverSat multimodal Earth-observation encoder into
[MMSegmentation 1.x](https://github.com/open-mmlab/mmsegmentation) as an
external project under `mmsegmentation/projects/universat/`.

## Layout

```
projects/universat/
├── universat/                         # Python package
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── backbones/
│   │   │   ├── __init__.py
│   │   │   ├── universat_backbone.py  # MMSeg 1.x backbone wrapper
│   │   │   └── universat_modules/     # Original encoder code (copied)
│   │   │       ├── UniverSat.py
│   │   │       ├── UniversalPatchEncoder.py
│   │   │       ├── modality_registry.py
│   │   │       ├── masking/
│   │   │       └── utils/
│   │   ├── decode_heads/
│   │   │   ├── universat_seg_head.py
│   │   │   └── universat_lp_head.py
│   │   └── data_preprocessors.py
│   └── datasets/
│       ├── __init__.py
│       ├── universat_dataset.py
│       └── transforms.py
├── configs/
│   ├── base_universat_seg.py                     # Generic segmentation template
│   ├── dataset/pastisr.py                        # Example PASTIS-R dataset config
│   ├── dataset/potsdam.py                        # Example Potsdam dataset config
│   ├── dataset/dior.py                           # DIOR dataset config (bbox -> mask)
│   ├── pastisr_universat-base_seg.py             # PASTIS-R segmentation experiment
│   ├── pastisr_universat-base_lp.py              # PASTIS-R linear-probe experiment
│   ├── extract_embeddings_pastisr_universat-base.py  # PASTIS-R extraction config
│   ├── extract_embeddings_potsdam_universat-base.py  # Potsdam extraction config
│   └── extract_embeddings_dior_universat-base.py     # DIOR extraction config
├── pastis/                            # Standalone PASTIS-R project
│   ├── universat_pastis/
│   ├── configs/
│   │   ├── universat-base_pastis_lp.py  # PASTIS-R linear probe
│   │   └── universat-base_pastis_ft.py  # PASTIS-R fine-tune
│   ├── train.sh
│   └── test.sh
├── train.sh
└── test.sh
```

## Requirements

- MMSegmentation 1.x / OpenMMLab 2.0 (uses `mmengine` + `mmseg.registry`)
- PyTorch >= 2.0 (the original code uses `torch.compile`)
- `safetensors` (for loading released `.safetensors` checkpoints)
- `einops` (used by `flexiVit.py`)

## PASTIS-R downstream evaluation

For a complete PASTIS-R linear-probe / fine-tuning project that follows the
same layout as `projects/copernicus/pastis`, see the `pastis/` subdirectory.
It contains a dedicated dataset class (`UniverSatPASTISDataset`), a custom
collate function for variable-length time series, and ready-to-use configs.

## Usage

### 1. Prepare data

Create a JSON split file for your dataset::

```json
[
  {
    "filenames": {
      "s2": "s2/xxx.npy",
      "s1": "s1/xxx.npy"
    },
    "ann": {"seg_map": "masks/xxx.png"},
    "height": 360,
    "width": 360
  }
]
```

Update `configs/dataset/pastisr.py` with your `data_root`, `split` paths,
`num_classes`, `ignore_index`, and per-modality `mean`/`std` statistics.

### 2. Prepare checkpoint

Put the pretrained UniverSat checkpoint (`.safetensors` or `.pth`) in the path
referenced by `MM_ARCHIVE_CKPT_HOME`, and make sure the config's
`init_cfg.checkpoint` points to it.

### 3. Train

From the MMSegmentation root (the folder containing `tools/`):

```bash
export PYTHONPATH=".:$PYTHONPATH"
python tools/train.py \
    projects/universat/configs/pastisr_universat-base_seg.py \
    --work-dir work_dirs/pastisr_universat-base_seg
```

Or use the provided launcher::

```bash
cd projects/universat
bash train.sh
```

### 4. Test

```bash
export PYTHONPATH=".:$PYTHONPATH"
python tools/test.py \
    projects/universat/configs/pastisr_universat-base_seg.py \
    path/to/checkpoint.pth \
    --work-dir work_dirs/pastisr_universat-base_seg/test
```

## Key components

- `UniverSatBackbone`: registered as `MODELS`, wraps the original encoder and
  exposes MMSeg-style multi-scale features.
- `UniverSatSegHead`: small conv-based segmentation head.
- `UniverSatLinearProbeHead`: LayerNorm + 1x1 classifier for linear probing.
- `UniverSatDataPreprocessor`: passes the multimodal dict through to the
  backbone.
- `UniverSatSegDataset` + `LoadMultimodalFromFile`/`NormalizeMultimodal`/
  `PackUniverSatInputs`: multimodal data loading for MMSeg 1.x.
- `UniverSatDIORDataset`: DIOR detection-to-mask wrapper for embedding
  extraction.

## Adapting to your own dataset

1. Add your modality names to `modalities` in both the dataset config and the
   backbone config.
2. If a modality is not in `modality_registry.py`, provide `wavelengths`,
   `input_res`, and `subpatches` overrides in the backbone config.
3. Replace `mean`/`std` in the dataset config with values computed from your
   training split.
4. Adjust `output_grid`, `patch_size`, and `crop_size` so they are consistent
   with your input patch layout.

## DIOR embedding extraction

DIOR is a detection dataset. To extract UniverSat backbone embeddings from its
RGB images within the MMSegmentation project, use
`configs/extract_embeddings_dior_universat-base.py` together with
`tools/extract_embeddings.py`. `UniverSatDIORDataset` converts each image's
bounding boxes to a filled semantic mask; the mask is saved as `label.tif` for
reference but no segmentation head is involved.

```bash
export PYTHONPATH=".:$PWD/projects/universat:$PYTHONPATH"
python projects/universat/tools/extract_embeddings.py \
    projects/universat/configs/extract_embeddings_dior_universat-base.py \
    --output-root work_dirs/universat_dior_embeddings \
    --splits train val test
```
