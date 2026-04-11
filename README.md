# Meshy AI Bulk Pipeline

Automates the full **Image -> 3D -> Remesh -> Retexture** workflow on [Meshy.ai](https://meshy.ai) so you can process a folder of images in one command instead of clicking through each model manually.

## What it does

For every `.jpg` / `.jpeg` / `.png` in your input folder the script will:

1. **Image to 3D** — generate a 3D mesh.
	- `single` mode: image-only generation (`should_remesh=false`, `should_texture=false`) followed by separate steps below.
	- `multi` mode: each group is submitted to Multi-Image-to-3D with `origin_at=bottom` (via `auto_size=true`) and no embedded remesh/texture.
2. **Remesh** — decimate to your target poly count (default 3 000 triangles) and resize to final height (default 0.05m).
	- In `multi` mode, remesh output is forced to **FBX only**.
3. **Retexture** — paint the remeshed model using the original image (Meshy 6, remove lighting, no PBR). *(single mode only; skipped in multi mode)*
4. **Download** — save the final model files to an output folder.

Progress is shown live in the terminal and state is saved to `pipeline_state.json` so you can resume if the script is interrupted.

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
# Basic — process all images in ./images, save to ./output
python meshy_pipeline.py -i ./images -o ./output

# Custom poly count and multiple output formats
python meshy_pipeline.py -i ./images -o ./output --polycount 5000 --formats glb fbx obj

# Use Meshy 5 instead of 6
python meshy_pipeline.py -i ./images -o ./output --ai-model meshy-5

# Use Multi-Image to 3D (automatically batched in groups of up to 4 images).
# Multi mode outputs FBX only, remeshed to 0.05m, with origin at bottom.
python meshy_pipeline.py -i ./images -o ./output --image-to-3d-mode multi

# Resume / re-download a previous run
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
| `--no-image-enhancement` | off | Disable image pre-processing |
| `--image-to-3d-mode` | `single` | `single` (POST `/image-to-3d`) or `multi` (POST `/multi-image-to-3d`) |
| `--multi-image-group-size` | `4` | Max images per multi-image task (`1` to `4`, only used in `multi` mode) |
| `--polycount` | `3000` | Target polygon count for remesh |
| `--resize-height` | `0.05` | Final model height in meters for remesh output |
| `--topology` | `triangle` | `triangle` or `quad` |
| `--formats` | `fbx` | Output formats (glb, fbx, obj, usdz, stl). In `multi` mode this is forced to `fbx`. |
| `--enable-pbr` | off | Generate PBR maps (metallic, roughness, normal) |
| `--no-remove-lighting` | off | Keep baked lighting in texture |
| `--poll-interval` | `10` | Seconds between status checks |
| `--submit-delay` | `1.0` | Seconds between task submissions |
| `--download-only` | off | Skip processing, just download |

## How it handles failures

- Each image is tracked independently — one failure won't stop the rest.
- State is saved after every submission and poll cycle to `<output>/pipeline_state.json`.
- Re-running the same command picks up where it left off (already-succeeded steps are skipped).
- The Meshy API rate limit (HTTP 429) is handled with automatic exponential backoff.

## Credit costs

Per image (approximate, based on Meshy pricing):

| Step | Credits |
|---|---|
| Image to 3D (no texture) | 20 |
| Remesh | 0 |
| Retexture (image-guided) | 10 |
| **Total per image** | **~30** |

Check your balance at the [Meshy API settings page](https://www.meshy.ai/settings/api).

## Notes

- The "License: private" setting you use in the web UI is an account-level default — the API always uses your account's license setting.
- Images are sent as base64 data URIs so they don't need to be publicly hosted.
- Multi-image mode uses `origin_at=bottom` during generation and then remeshes to `0.05m` height with FBX output.
- In multi mode, standalone retexture is skipped.
- In multi-image mode, each group becomes one task and output files are named from the first image in the group (with `__multiN` suffix).
- Very large images (>10 MB) may be slow to upload; consider resizing first.
