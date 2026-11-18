# Third-party notices

The MIT License in [LICENSE](LICENSE) applies to the code and documentation written for this project.
The following third-party components are **not** covered by it and are **not included** in this repository:

| Component | License | How it is used |
|---|---|---|
| [SAM 2 / SAM 2.1](https://github.com/facebookresearch/sam2) code and checkpoints (Meta) | Apache License 2.0 | Installed separately (`backend/third_party/sam2`, `backend/checkpoints`) |
| [SA-V dataset](https://ai.meta.com/datasets/segment-anything-video/) video `sav_000001` and its annotations (Meta) | CC BY 4.0 | Frames and tracking results shown in `docs/images` and `docs/realworld` |
| [FFmpeg](https://ffmpeg.org/) | LGPL / GPL (depending on build) | Downloaded locally by `backend/scripts/setup_ffmpeg.py`, not redistributed |

Demo videos from the SAM 2 repository that do not state their own license were used only for local evaluation;
images and videos derived from them are excluded from this repository (see `.gitignore`).
