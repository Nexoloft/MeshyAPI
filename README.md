# Meshy AI Bulk Pipeline

Automates batch model generation on [Meshy.ai](https://meshy.ai) from a folder of input images.

## Default behavior

Running this command:

```bash
python meshy_pipeline.py -i ./images -o ./output
```

processes each image independently via **Image to 3D API** (`POST /image-to-3d`) using:

- `should_texture=true`
- `should_remesh=true`
- `target_polycount=3000`
- `topology=triangle`
- `enable_pbr=false`
- `remove_lighting=true`
- `image_enhancement=true`
- `target_formats=["fbx"]`
- `auto_size=false`
- `origin_at=bottom`

The pipeline then downloads the generated FBX files into `<output>/models`.

## Optional exact-size mode

If you want exact output height (for example, `0.05m`), run an extra remesh step:

```bash
python meshy_pipeline.py -i ./images -o ./output --enforce-height 0.05
```

When `--enforce-height` is set, the pipeline remeshes each successful Image-to-3D result with:

- `resize_height=<enforce-height>`
- `origin_at=bottom`
- `target_formats=["fbx"]`

## Setup

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and paste your [Meshy API key](https://www.meshy.ai/settings/api):

```bash
cp .env.example .env
# edit .env and add your key
```

## Usage

```bash
# Default (recommended): one Image-to-3D task per image
python meshy_pipeline.py -i ./images -o ./output

# Adjust polycount while keeping all other defaults
python meshy_pipeline.py -i ./images -o ./output --polycount 5000

# Enforce exact output height via post-remesh
python meshy_pipeline.py -i ./images -o ./output --enforce-height 0.05

# Re-download results from saved state
python meshy_pipeline.py -i ./images -o ./output --download-only
```

### All flags

| Flag | Default | Description |
|---|---|---|
| `-i`, `--input` | *(required)* | Folder with source images |
| `-o`, `--output` | `./output` | Where to save state and models |
| `--api-key` | `MESHY_API_KEY` env | Your Meshy API key |
| `--ai-model` | `meshy-6` | `meshy-5`, `meshy-6`, or `latest` |
| `--model-type` | `standard` | `standard` or `lowpoly` |
| `--no-image-enhancement` | off | Disable image enhancement |
| `--polycount` | `3000` | Target polygon count for remesh-in-generation |
| `--enforce-height` | unset | Optional exact output height in meters via post-remesh |
| `--topology` | `triangle` | `triangle` or `quad` |
| `--formats` | `fbx` | Accepted for compatibility; pipeline enforces FBX output |
| `--enable-pbr` | off | Generate PBR maps (disabled by default) |
| `--no-remove-lighting` | off | Keep baked lighting (default removes lighting) |
| `--resize-height` | `0.05` | Deprecated alias for `--enforce-height` |
| `--poll-interval` | `10` | Seconds between status checks |
| `--submit-delay` | `1.0` | Seconds between task submissions |
| `--download-only` | off | Skip processing and only download from saved state |

## Failure handling

- Each input image is tracked independently.
- State is saved to `<output>/pipeline_state.json`.
- Re-running the same command resumes from previously saved state.
- HTTP 429 responses are retried with exponential backoff.
