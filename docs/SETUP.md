# Setup and usage

## Download required files

These files are **not** in this repository. Each script checks for them at
startup and stops with a message if one is missing.

### 1. Python environment

GNM Head is tested with Python 3.13.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. GNM Head (Google, Apache 2.0)

GNM Head is released by Google under the Apache 2.0 license and is not
redistributed here. Get it from the official sources:

- Code: GitHub, <https://github.com/google/GNM>
- Model weights: `gnm_head.npz`, version 3.0, from either
  - Hugging Face: <https://huggingface.co/google/gnm-v3> (file `v3_0/gnm_head.npz`)
  - Kaggle: <https://www.kaggle.com/models/google/gnm-v3>

**a) Install the GNM code** into the project root (the `gnm/` folder is git-ignored):

```bash
git clone https://github.com/google/GNM.git gnm
git -C gnm checkout 3ec07cb   # tested commit, see note below
pip install -e "gnm/gnm/shape[pytorch]"
```

Tested with GNM commit `3ec07cb`. Newer GNM releases may remove `GNM.from_local()`,
which these scripts use. With `3ec07cb` it works but prints a `DeprecationWarning`,
which you can ignore.

**b) Put the weights in the GNM cache folder.** The scripts load the model with
`GNM.from_local()`, which reads `v3_0/gnm_head.npz` from GNM's cache folder:

| Setting | Weights path |
| --- | --- |
| default | `~/.cache/gnm/models/v3_0/gnm_head.npz` |
| `XDG_CACHE_HOME` set | `$XDG_CACHE_HOME/gnm/models/v3_0/gnm_head.npz` |
| `GNM_CACHE_DIR` set | `$GNM_CACHE_DIR/v3_0/gnm_head.npz` |

```bash
mkdir -p ~/.cache/gnm/models/v3_0
mv ~/Downloads/gnm_head.npz ~/.cache/gnm/models/v3_0/
```

The scripts check for this file at startup and will not download it for you.

**c) Check the file.** The SHA-256 must be
`61d78bbfb4ad8e0b38495804a4caef3214d3df00f8c3f68761e63b41ce3747eb`
(Hugging Face release, 53,328,601 bytes):

```bash
shasum -a 256 ~/.cache/gnm/models/v3_0/gnm_head.npz   # macOS
sha256sum     ~/.cache/gnm/models/v3_0/gnm_head.npz   # Linux
```

> **Older GNM clones.** GitHub clones from before 2026-09-29 include the weights at
> `gnm/gnm/shape/data/versions/v3_0/gnm_head.npz`, and `from_local()` loads that copy.
> Its SHA-256 is different (`55bdf0c22e6462fe82ff85f702a8cbe1e83135351b161f55234bee0b7041e7d4`)
> because the `.npz` archive is packed differently, but its 23 arrays are identical
> to the Hugging Face file. The startup check accepts either location.

### 3. MediaPipe Face Landmarker model

```bash
curl -L -o models/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

### Expected layout

```
GNM_Avatar/
├── data/head_dense_468_v2_nowarp.txt  # default correspondence, in this repo
├── data/head_dense_468_v2.txt         # v2, for comparison, in this repo
├── gnm/                        # git clone of google/GNM (ignored)
├── models/face_landmarker.task # MediaPipe model (ignored)
└── src/

~/.cache/gnm/models/
└── v3_0/gnm_head.npz           # GNM Head v3.0 weights (or $GNM_CACHE_DIR/v3_0/)
```

## Usage

Run from the project root:

```bash
python src/fit_gnm.py --image <your_photo>.jpg          # fit
python src/fit_gnm.py --image <your_photo>.jpg --view   # fit + 3D viewer
```

Optional flags (defaults keep the behaviour above):

| Flag | Default | Purpose |
| --- | --- | --- |
| `--landmarks <file>` | `data/head_dense_468_v2_nowarp.txt` | Correspondence file to fit with |
| `--out-dir <folder>` | `data/` | Where the outputs are written |
| `--metrics-json <file>` | off | Also write errors, clamp counts and fit time as JSON |

Outputs (in `data/` by default, git-ignored):

| File | Content |
| --- | --- |
| `gnm_fit.obj` | Fitted head mesh |
| `gnm_fit.npz` | Identity, expression, rotations, translation, camera scale/offset |
| `gnm_fit_diagnostic.png` | MediaPipe vs projected GNM landmarks with error lines |
| `gnm_fit_mesh_overlay.png` | Fitted mesh drawn over the photo |

Other scripts in `src/`:

| Script | Purpose |
| --- | --- |
| `gnm_mediapipe_landmarks.py` | Loader for the 468-point correspondence + self-check on the GNM template |
| `test_correspondence.py` | Evaluates the correspondence on the neutral GNM head |
| `gnm_inspect.py` | Prints GNM dimensions and plots its 68 sparse landmarks |
| `gnm_test.py` | Shows a sampled "happy" expression |
| `mediapipe_test.py` | Draws MediaPipe landmarks on an image: `--image <your_photo>.jpg` |
| `batch_fit.py` | Fits every `.jpg` in a folder with the three correspondence versions, writes `results/metrics.csv`, `results/metrics.md`, `results/summary_grid.png` and `results/ear_check.png` |
| `identity_check.py` | Compares fitted identities with the GNM mean head, each other and random GNM identities; writes the identity section of `results/metrics.md` |
| `render_3d.py` | Renders the fitted meshes in grey (front, 3/4, side) to `results/<version>/summary_grid_3d.png` |
| `render_pipeline.py` | Draws the README pipeline figure `docs/pipeline.png` from one photo's results (default P3, v2-nowarp) |
| `preflight.py` | Startup check for the files above |

## Privacy details

Do not commit photos of people or fitted meshes of them. `.gitignore` excludes all
images, `.obj` and `.npz` files, everything in `data/` except the two
correspondence files and `SOURCES.md`, and the per-photo folders and photo grids in
`results/` (they contain the photos and their landmark coordinates).

Published results use the labels P1–P8 only: `results/metrics.md`,
`results/metrics.csv`, `results/v2-nowarp/summary_grid_3d.png` and `docs/pipeline.png`.
The two figures are the only published files with the photos: a face crop of each
Pixabay photo next to its 3D render, and the P3 crop in the pipeline figure. The
crops exist only inside those PNGs.
[`data/SOURCES.md`](../data/SOURCES.md) lists each label's Pixabay page,
contributor and license.
