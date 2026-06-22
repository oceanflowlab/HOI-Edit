# Assets

Large assets are not committed to GitHub. They should be downloaded separately and placed under the repository paths below.

## Baidu Netdisk

Download:

- Link: https://pan.baidu.com/s/1XBGmKSqbz_j8q2Tjkxaghg
- Extraction code: `vx45`

SHA256:

```text
387f10b42883e6a00539064389c2adcb74b5e438ea53373cae83f5a5a5bfcb8e  hoi_edit_assets_v7.tar.gz
```

## Contents

| Asset | Size | Count / file | Expected path after extraction |
|---|---:|---:|---|
| HOI-Edit annotations | small | 2 JSON files | `data/` |
| HOI-Edit L1/L2 original images | 476 MB | 499 images | `data/data_v7_L12/` |
| HOI-Edit L3 original images | 126 MB | 143 images | `data/data_v7_L3/` |
| GroundingDINO weight | 694 MB | `groundingdino_swint_ogc.pth` | `weights/` |
| SAM2 checkpoint | 898 MB | `sam2.1_hiera_large.pt` | `weights/` |

Checkpoint SHA256:

```text
3b3ca2563c77c69f651d7bd133e97139c186df06231157a64c507099c52bc799  groundingdino_swint_ogc.pth
2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318  sam2.1_hiera_large.pt
```

## Notes

- The GitHub repository keeps code and annotation JSON files only.
- `data/data_v7_L12/`, `data/data_v7_L3/`, `data/*_frames/`, and `weights/` are ignored by git.
- Download `hoi_edit_assets_v7.tar.gz` and extract it at the repository root:

```bash
tar -xzf hoi_edit_assets_v7.tar.gz
```
