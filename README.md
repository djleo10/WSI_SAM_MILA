# SAM-Guided Patch Sampling for Whole-Slide Pathology Images

Code from my research collaboration at **Mila — Québec AI Institute** (summer 2024, Prof. Samira Ebrahimi Kahou): using Meta's **Segment Anything Model (SAM)** to pick diagnostically relevant regions in gigapixel whole-slide images (WSIs) instead of tiling them on a fixed grid.

## Why

A single lymph-node WSI can be ~100,000 × 200,000 pixels. Fixed-grid tiling produces huge numbers of patches, many of them background or uninformative. Mask-guided sampling keeps the tissue that matters and cuts compute and memory.

## Pipeline (`t2v-sam-final-prev.ipynb`)

1. **Load slides** from CAMELYON17 with [TIAToolbox](https://github.com/TissueImageAnalytics/tiatoolbox) and inspect their resolution pyramid
2. **Tissue masking** — thresholding (Otsu) and morphological filtering to separate tissue from background, with hole counting to reject poor regions
3. **Patch extraction** — crop to the patch grid and save 256 × 256 patches with their masks
4. **Segment Anything** — load SAM (ViT-H) and generate masks to guide which regions are sampled

These patches feed **multiple-instance learning (MIL)** slide classifiers.

## Results (from the full project)

- About **50 % fewer patches and less memory** than fixed-grid sampling on CAMELYON17, with diagnostic accuracy preserved
- Slide-level **AUC 0.97 (+4 % over baseline)** when adaptive sampling was scaled up

## Requirements

Python 3.10, PyTorch with CUDA, `tiatoolbox`, `segment-anything`, `patchify`, OpenCV. The notebook was run on Kaggle with the CAMELYON17 dataset.
