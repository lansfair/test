# UniverSat Embedding Extraction Tools

This directory contains tools for extracting dense backbone embeddings from the
UniverSat model on PASTIS-R, following the same manifest/GeoTIFF layout as
`olmoearth/tools/extract_embeddings.py`.

## Files

- `extract_embeddings.py` — main extraction script.
- `common.py` — shared helpers (GeoTIFF I/O, JSON manifest, progress, affine math).
- `extract_embeddings_pastisr.sh` — example launch script for PASTIS-R.
- `extract_embeddings_potsdam.sh` — example launch script for Potsdam.
- `extract_embeddings_dior.sh` — example launch script for DIOR.

## Usage

### PASTIS-R

From the MMSegmentation root (the directory containing `tools/`):

```bash
export PYTHONPATH=".:$PWD/projects/universat:$PWD/projects/universat/pastis:$PYTHONPATH"
export MM_ARCHIVE_DATA_HOME=/path/to/data
export MM_ARCHIVE_CKPT_HOME=/path/to/checkpoints

python projects/universat/tools/extract_embeddings.py \
    projects/universat/configs/extract_embeddings_pastisr_universat-base.py \
    --output-root work_dirs/universat_pastisr_embeddings \
    --splits train val test \
    --batch-size 1 \
    --tile-size 0 \
    --device auto \
    --precision bf16
```

Or use the provided shell script:

```bash
bash projects/universat/tools/extract_embeddings_pastisr.sh
```

### DIOR

Make sure the DIOR release follows the standard layout and that the
environment variables are set:

```bash
export PYTHONPATH=".:$PWD/projects/universat:$PYTHONPATH"
export MM_ARCHIVE_DATA_HOME=/path/to/data
export MM_ARCHIVE_CKPT_HOME=/path/to/checkpoints

python projects/universat/tools/extract_embeddings.py \
    projects/universat/configs/extract_embeddings_dior_universat-base.py \
    --output-root work_dirs/universat_dior_embeddings \
    --splits train val test \
    --batch-size 1 \
    --tile-size 0 \
    --device auto \
    --precision bf16
```

Or use the provided shell script:

```bash
bash projects/universat/tools/extract_embeddings_dior.sh
```

For distributed extraction:

```bash
torchrun --nproc_per_node=2 projects/universat/tools/extract_embeddings.py \
    projects/universat/configs/extract_embeddings_pastisr_universat-base.py \
    --output-root work_dirs/universat_pastisr_embeddings \
    --splits train val test
```

## Output Layout

```
output_root/
  train/
    <sample_id>/
      embedding.tif   # CHW float32 feature map
      label.tif       # HW uint8/int32 label map
      input_s2.tif    # (optional) time-mean S2 input
      input_s1.tif    # (optional) time-mean S1 input
  train.json          # manifest listing all train samples
  val.json
  test.json
  summary.json        # run summary
```

## Notes

- The DIOR pipeline is RGB-only. DIOR is a detection dataset, so
  `UniverSatDIORDataset` converts each image's bounding boxes to a filled
  semantic mask. The mask is saved as `label.tif` for reference but is not used
  to train any head during extraction.
- The PASTIS-R pipeline packs inputs as a dict `{s2, s1, s2_dates, s1_dates}`.
  The extraction script stacks and forwards this dict directly to
  `UniverSatBackbone`.
- `--save-raw-inputs` is not supported for PASTIS-R because the dataset does not
  expose pre-normalization arrays through the sample metainfo.
- `--save-inputs` saves a time-mean of each 4-D modality tensor as a multi-band
  GeoTIFF (`input_<modality>.tif`).
- Default `--tile-size 0` disables sliding-window extraction, which is correct
  for the native 128×128 PASTIS-R patches. Enable tiling only if you pad/crop
  inputs to larger sizes.
