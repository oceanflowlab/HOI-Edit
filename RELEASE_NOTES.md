# HOI-Edit / HOI-Eval / SCPE Release Notes

This repository is a lightweight open-source release assembled from the portable HOI-Eval evaluation package and the SCPE / ACE I2V pipeline.

## Included

- SCPE / ACE I2V scripts under `scpe/scripts/`
- SCPE prompt templates, QA2 prompts, Wan2.2 wrappers, and Playbook seeds under `scpe/data/`
- SCPE local setup templates and runner scripts under `scpe/`
- Evaluation entry scripts: `run_qa_hoi.sh` and `run_eval.sh`
- HOI-Edit annotation JSON files under `data_v7/CR/`
- QA, HOI, resizing, scoring, and path utility scripts under `evaluation/`
- SAM2 source code needed by the evaluation wrapper
- GroundingDINO source code needed by the evaluation wrapper
- Environment requirement files and local configuration template under `env/`

## Not Included

Large or user-specific artifacts are intentionally not committed:

- Original CR images: `data_v7/CR/data_v7_L12/`, `data_v7/CR/data_v7_L3/`
- Edited frames: `data_v7/CR/<model>_frames/`
- GroundingDINO weights: `third_party/GroundingDINO/weights/`
- SAM2 checkpoints: `sam2/checkpoints/`
- SCPE outputs: `scpe/output/`, `scpe/logs/`
- SCPE local credentials and paths: `scpe/env.local`
- Wan2.2 source checkout and weights
- Runtime outputs under `eval_runs/`
- Local credentials and paths in `env/local.conf`

Use `env/local.conf.example` and `scpe/env.example` as templates for local paths and API keys.

## Source Integration Note

The release was prepared from the available lite tarball:

```text
cr_eval_release_v7_lite.tar.gz
```

The SCPE component was integrated from the available local source:

```text
ace_i2v_basic
```

The additional requested source path:

```text
/network_space/server127_2/shared/gjy/cr_eval_release
```

was not mounted or readable in the current environment at packaging time. The packaged scripts already use `env/workspace.conf` to resolve paths relative to the repository root, so no algorithmic code changes were made.
