# Third-Party Notices

This release vendors code from the following third-party projects for reproducible evaluation setup:

- GroundingDINO, included under `third_party/GroundingDINO/`. See `third_party/GroundingDINO/LICENSE` and `third_party/GroundingDINO/README.md`.
- SAM2-related source code, included under `sam2/sam2/`. Please verify upstream licensing requirements before redistributing beyond this temporary release.
- SCPE scripts call Google Gemini through `google-genai` and can call DashScope Wan2.2 I2V through the `dashscope` SDK when configured by the user.
- Local Wan2.2 generation expects a separate Wan2.2 checkout and weights supplied by the user; those assets are not vendored here.

Large pretrained model weights and checkpoints are not included in this repository. Download and place them locally as described in `README.md`.
