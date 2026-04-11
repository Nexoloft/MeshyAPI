#!/usr/bin/env python3
"""
Meshy AI Bulk Pipeline
Automates: Image to 3D -> Remesh -> Retexture workflow in batch.
"""

import argparse
import base64
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

import requests
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
    TaskProgressColumn,
)
from rich.table import Table
from rich.panel import Panel

load_dotenv()

BASE_URL = "https://api.meshy.ai/openapi/v1"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
STATE_FILENAME = "pipeline_state.json"

console = Console()


# ---------------------------------------------------------------------------
# Data model – one entry per source image
# ---------------------------------------------------------------------------

@dataclass
class ImageTask:
    image_path: str
    input_images: list[str] = field(default_factory=list)
    image_to_3d_endpoint: str = "image-to-3d"
    image_to_3d_id: Optional[str] = None
    image_to_3d_status: str = "NOT_STARTED"
    remesh_id: Optional[str] = None
    remesh_status: str = "NOT_STARTED"
    retexture_id: Optional[str] = None
    retexture_status: str = "NOT_STARTED"
    final_model_urls: Optional[dict] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Thin API client with automatic retry on 429
# ---------------------------------------------------------------------------

class MeshyClient:
    def __init__(self, api_key: str, max_retries: int = 6):
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        })
        self.max_retries = max_retries

    def _request(self, method: str, path: str, **kwargs) -> Any:
        url = f"{BASE_URL}{path}"
        for attempt in range(self.max_retries):
            resp = self.session.request(method, url, **kwargs)
            if resp.status_code == 429:
                wait = min(2 ** attempt * 5, 120)
                console.print(f"  [yellow]Rate-limited – retrying in {wait}s …[/]")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json() if resp.content else None
        raise RuntimeError(f"Still rate-limited after {self.max_retries} retries")

    # -- Image to 3D ---------------------------------------------------------

    def create_image_to_3d(self, image_data_uri: str, cfg: dict) -> str:
        payload = {
            "image_url": image_data_uri,
            "model_type": cfg["model_type"],
            "ai_model": cfg["ai_model"],
            "image_enhancement": cfg["image_enhancement"],
            "pose_mode": cfg["pose_mode"],
            "should_texture": False,
            "should_remesh": False,
        }
        result = self._request("POST", "/image-to-3d", json=payload)
        if not isinstance(result, dict) or "result" not in result:
            raise RuntimeError("Unexpected response for create_image_to_3d")
        return str(result["result"])

    def get_image_to_3d(self, task_id: str) -> dict:
        result = self._request("GET", f"/image-to-3d/{task_id}")
        if not isinstance(result, dict):
            raise RuntimeError("Unexpected response for get_image_to_3d")
        return result

    def create_multi_image_to_3d(self, image_data_uris: list[str], cfg: dict) -> str:
        payload = {
            "image_urls": image_data_uris,
            "ai_model": cfg["ai_model"],
            "image_enhancement": cfg["image_enhancement"],
            "pose_mode": cfg["pose_mode"],
            "should_texture": False,
            "should_remesh": False,
            "target_formats": cfg["target_formats"],
            "auto_size": True,
            "origin_at": "bottom",
        }
        result = self._request("POST", "/multi-image-to-3d", json=payload)
        if not isinstance(result, dict) or "result" not in result:
            raise RuntimeError("Unexpected response for create_multi_image_to_3d")
        return str(result["result"])

    def get_multi_image_to_3d(self, task_id: str) -> dict:
        result = self._request("GET", f"/multi-image-to-3d/{task_id}")
        if not isinstance(result, dict):
            raise RuntimeError("Unexpected response for get_multi_image_to_3d")
        return result

    # -- Remesh ---------------------------------------------------------------

    def create_remesh(self, input_task_id: str, cfg: dict) -> str:
        payload = {
            "input_task_id": input_task_id,
            "target_polycount": cfg["target_polycount"],
            "resize_height": cfg["resize_height"],
            "topology": cfg["topology"],
            "target_formats": cfg["target_formats"],
        }
        result = self._request("POST", "/remesh", json=payload)
        if not isinstance(result, dict) or "result" not in result:
            raise RuntimeError("Unexpected response for create_remesh")
        return str(result["result"])

    def get_remesh(self, task_id: str) -> dict:
        result = self._request("GET", f"/remesh/{task_id}")
        if not isinstance(result, dict):
            raise RuntimeError("Unexpected response for get_remesh")
        return result

    # -- Retexture ------------------------------------------------------------

    def create_retexture(self, input_task_id: str, image_data_uri: str, cfg: dict) -> str:
        payload = {
            "input_task_id": input_task_id,
            "image_style_url": image_data_uri,
            "ai_model": cfg["retexture_ai_model"],
            "remove_lighting": cfg["remove_lighting"],
            "enable_pbr": cfg["enable_pbr"],
            "enable_original_uv": cfg["enable_original_uv"],
        }
        result = self._request("POST", "/retexture", json=payload)
        if not isinstance(result, dict) or "result" not in result:
            raise RuntimeError("Unexpected response for create_retexture")
        return str(result["result"])

    def get_retexture(self, task_id: str) -> dict:
        result = self._request("GET", f"/retexture/{task_id}")
        if not isinstance(result, dict):
            raise RuntimeError("Unexpected response for get_retexture")
        return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def image_to_data_uri(path: str) -> str:
    ext = Path(path).suffix.lower()
    mime = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"
    raw = Path(path).read_bytes()
    return f"data:{mime};base64,{base64.b64encode(raw).decode()}"


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------

class Pipeline:
    def __init__(self, client: MeshyClient, input_folder: str,
                 output_folder: str, cfg: dict):
        self.client = client
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.cfg = cfg
        self.state_file = self.output_folder / STATE_FILENAME
        self.tasks: list[ImageTask] = []

    # -- persistence ----------------------------------------------------------

    def save_state(self):
        self.output_folder.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, "w") as f:
            json.dump([asdict(t) for t in self.tasks], f, indent=2)

    def load_state(self) -> dict[str, dict]:
        if not self.state_file.exists():
            return {}
        with open(self.state_file) as f:
            return {entry["image_path"]: entry for entry in json.load(f)}

    # -- discovery ------------------------------------------------------------

    def discover_images(self):
        images = sorted(
            p for p in self.input_folder.iterdir()
            if p.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if not images:
            console.print(f"[red]No images (.jpg/.jpeg/.png) found in {self.input_folder}[/]")
            sys.exit(1)

        existing = self.load_state()
        if self.cfg["image_to_3d_mode"] == "multi":
            group_size = self.cfg["multi_image_group_size"]
            groups = [images[i:i + group_size] for i in range(0, len(images), group_size)]

            for group in groups:
                group_paths = [str(p) for p in group]
                key = "||".join(group_paths)
                if key in existing:
                    task = ImageTask(**existing[key])
                    if not task.input_images:
                        task.input_images = group_paths
                    task.image_to_3d_endpoint = "multi-image-to-3d"
                    # Normalize legacy state created before explicit multi remesh flow.
                    if task.remesh_status == "SUCCEEDED" and not task.remesh_id:
                        task.remesh_status = "NOT_STARTED"
                    if task.retexture_status == "SUCCEEDED" and not task.retexture_id:
                        task.retexture_status = "NOT_STARTED"
                    self.tasks.append(task)
                else:
                    self.tasks.append(ImageTask(
                        image_path=key,
                        input_images=group_paths,
                        image_to_3d_endpoint="multi-image-to-3d",
                    ))

            resumed = sum(1 for t in self.tasks if t.image_to_3d_status != "NOT_STARTED")
            console.print(
                f"[bold]{len(images)}[/] images found in [bold]{len(self.tasks)}[/] multi-image groups"
                + (f" ({resumed} resumed from previous run)" if resumed else "")
            )
            return

        for img in images:
            key = str(img)
            if key in existing:
                task = ImageTask(**existing[key])
                if not task.input_images:
                    task.input_images = [key]
                if task.image_to_3d_status == "NOT_STARTED":
                    task.image_to_3d_endpoint = "image-to-3d"
                self.tasks.append(task)
            else:
                self.tasks.append(ImageTask(
                    image_path=key,
                    input_images=[key],
                    image_to_3d_endpoint="image-to-3d",
                ))

        resumed = sum(1 for t in self.tasks if t.image_to_3d_status != "NOT_STARTED")
        console.print(
            f"[bold]{len(self.tasks)}[/] images found"
            + (f" ({resumed} resumed from previous run)" if resumed else "")
        )

    def _task_images(self, task: ImageTask) -> list[str]:
        return task.input_images or [task.image_path]

    def _task_name(self, task: ImageTask) -> str:
        images = self._task_images(task)
        primary = Path(images[0]).name
        if len(images) == 1:
            return primary
        return f"{primary} (+{len(images) - 1} more)"

    def _task_output_stem(self, task: ImageTask) -> str:
        images = self._task_images(task)
        stem = Path(images[0]).stem
        if len(images) > 1:
            return f"{stem}__multi{len(images)}"
        return stem

    # -- generic poll loop ----------------------------------------------------

    def _poll_until_done(self, label: str, get_fn, id_attr: str,
                         status_attr: str):
        active = [
            t for t in self.tasks
            if getattr(t, id_attr) is not None
            and getattr(t, status_attr) not in ("SUCCEEDED", "FAILED", "NOT_STARTED")
        ]
        if not active:
            return

        with Progress(
            SpinnerColumn(),
            TextColumn(f"[bold blue]{label}[/]"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            bar = progress.add_task(label, total=len(active))

            while active:
                for task in active[:]:
                    try:
                        result = get_fn(task, getattr(task, id_attr))
                        status = result.get("status", "UNKNOWN")
                        setattr(task, status_attr, status)

                        if status == "SUCCEEDED":
                            if "model_urls" in result:
                                task.final_model_urls = result["model_urls"]
                            active.remove(task)
                            progress.advance(bar)
                        elif status == "FAILED":
                            msg = result.get("task_error", {}).get("message", "Unknown")
                            task.error = f"{label} failed: {msg}"
                            active.remove(task)
                            progress.advance(bar)
                            console.print(f"  [red]FAILED[/] {self._task_name(task)}: {msg}")
                    except Exception as exc:
                        console.print(f"  [yellow]Poll error ({self._task_name(task)}): {exc}[/]")

                self.save_state()
                if active:
                    time.sleep(self.cfg["poll_interval"])

    # -- pipeline steps -------------------------------------------------------

    def _submit_batch(self, label: str, candidates: list[ImageTask],
                      submit_fn):
        if not candidates:
            return
        with Progress(
            SpinnerColumn(),
            TextColumn(f"[bold]Submitting {label}[/]"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            bar = progress.add_task("submit", total=len(candidates))
            for task in candidates:
                try:
                    submit_fn(task)
                except Exception as exc:
                    console.print(f"  [red]x[/] {self._task_name(task)}: {exc}")
                progress.advance(bar)
                self.save_state()
                time.sleep(self.cfg["submit_delay"])

    def step_image_to_3d(self):
        if self.cfg["image_to_3d_mode"] == "multi":
            console.rule("[bold green]Step 1/3 : Multi-Image to 3D (origin=bottom)")
        else:
            console.rule("[bold green]Step 1/3 : Image to 3D")
        need = [t for t in self.tasks if t.image_to_3d_status == "NOT_STARTED"]

        def submit(task: ImageTask):
            uris = [image_to_data_uri(path) for path in self._task_images(task)]
            if self.cfg["image_to_3d_mode"] == "multi":
                task.image_to_3d_endpoint = "multi-image-to-3d"
                tid = self.client.create_multi_image_to_3d(uris, self.cfg)
            else:
                task.image_to_3d_endpoint = "image-to-3d"
                tid = self.client.create_image_to_3d(uris[0], self.cfg)
            task.image_to_3d_id = tid
            task.image_to_3d_status = "PENDING"
            console.print(f"  [green]+[/] {self._task_name(task)} -> {tid}")

        def get_image_status(task: ImageTask, task_id: str) -> dict:
            if task.image_to_3d_endpoint == "multi-image-to-3d":
                return self.client.get_multi_image_to_3d(task_id)
            return self.client.get_image_to_3d(task_id)

        self._submit_batch("Image-to-3D tasks", need, submit)
        self._poll_until_done(
            "Waiting for Image-to-3D",
            get_image_status,
            "image_to_3d_id",
            "image_to_3d_status",
        )

    def step_remesh(self):
        console.rule("[bold green]Step 2/3 : Remesh to "
                     f"{self.cfg['target_polycount']} polys @ {self.cfg['resize_height']}m")
        need = [
            t for t in self.tasks
            if t.image_to_3d_status == "SUCCEEDED" and t.remesh_status == "NOT_STARTED"
        ]

        def submit(task: ImageTask):
            if not task.image_to_3d_id:
                raise RuntimeError("Missing Image-to-3D task id for remesh")
            tid = self.client.create_remesh(task.image_to_3d_id, self.cfg)
            task.remesh_id = tid
            task.remesh_status = "PENDING"
            console.print(f"  [green]+[/] {self._task_name(task)} -> {tid}")

        self._submit_batch("Remesh tasks", need, submit)
        self._poll_until_done(
            "Waiting for Remesh",
            lambda _task, task_id: self.client.get_remesh(task_id),
            "remesh_id",
            "remesh_status",
        )

    def step_retexture(self):
        if self.cfg["image_to_3d_mode"] == "multi":
            console.rule("[bold green]Step 3/3 : Retexture (skipped in Multi-Image mode)")
            for task in self.tasks:
                if task.remesh_status == "SUCCEEDED" and task.retexture_status == "NOT_STARTED":
                    task.retexture_status = "SKIPPED"
            self.save_state()
            return
        console.rule("[bold green]Step 3/3 : Retexture with original image")
        need = [
            t for t in self.tasks
            if t.remesh_status == "SUCCEEDED" and t.retexture_status == "NOT_STARTED"
        ]

        def submit(task: ImageTask):
            if not task.remesh_id:
                raise RuntimeError("Missing remesh task id for retexture")
            uri = image_to_data_uri(self._task_images(task)[0])
            tid = self.client.create_retexture(task.remesh_id, uri, self.cfg)
            task.retexture_id = tid
            task.retexture_status = "PENDING"
            console.print(f"  [green]+[/] {self._task_name(task)} -> {tid}")

        self._submit_batch("Retexture tasks", need, submit)
        self._poll_until_done(
            "Waiting for Retexture",
            lambda _task, task_id: self.client.get_retexture(task_id),
            "retexture_id",
            "retexture_status",
        )

    # -- download -------------------------------------------------------------

    def download_results(self):
        console.rule("[bold green]Downloading final models")
        models_dir = self.output_folder / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        if self.cfg["image_to_3d_mode"] == "multi":
            done = [t for t in self.tasks if t.remesh_status == "SUCCEEDED"]
        else:
            done = [t for t in self.tasks if t.retexture_status == "SUCCEEDED"]
        if not done:
            if self.cfg["image_to_3d_mode"] == "multi":
                console.print("[yellow]No completed remesh tasks to download.[/]")
            else:
                console.print("[yellow]No completed retexture tasks to download.[/]")
            return

        for task in done:
            if self.cfg["image_to_3d_mode"] == "multi":
                if not task.remesh_id:
                    console.print(f"  [red]Missing remesh id for {self._task_name(task)}[/]")
                    continue
                try:
                    result = self.client.get_remesh(task.remesh_id)
                    task.final_model_urls = result.get("model_urls", {})
                except Exception as exc:
                    console.print(f"  [red]Cannot fetch URLs for "
                                  f"{self._task_name(task)}: {exc}[/]")
                    continue
            elif not task.final_model_urls:
                try:
                    if task.retexture_id:
                        result = self.client.get_retexture(task.retexture_id)
                    elif task.image_to_3d_endpoint == "multi-image-to-3d" and task.image_to_3d_id:
                        result = self.client.get_multi_image_to_3d(task.image_to_3d_id)
                    elif task.image_to_3d_id:
                        result = self.client.get_image_to_3d(task.image_to_3d_id)
                    else:
                        console.print(f"  [red]Missing task id for {self._task_name(task)}[/]")
                        continue
                    task.final_model_urls = result.get("model_urls", {})
                except Exception as exc:
                    console.print(f"  [red]Cannot fetch URLs for "
                                  f"{self._task_name(task)}: {exc}[/]")
                    continue

            if not task.final_model_urls:
                console.print(f"  [yellow]No model URLs for {self._task_name(task)}[/]")
                continue

            stem = self._task_output_stem(task)
            for fmt, url in task.final_model_urls.items():
                if not url:
                    continue
                try:
                    resp = requests.get(url, timeout=120)
                    resp.raise_for_status()
                    out = models_dir / f"{stem}.{fmt}"
                    out.write_bytes(resp.content)
                    console.print(f"  [green]Saved[/] {out}")
                except Exception as exc:
                    console.print(f"  [red]Download failed[/] {stem}.{fmt}: {exc}")

        self.save_state()

    # -- summary --------------------------------------------------------------

    def print_summary(self):
        table = Table(title="Pipeline Summary", show_lines=True)
        table.add_column("Image", style="cyan", max_width=30)
        table.add_column("Image->3D", justify="center")
        table.add_column("Remesh", justify="center")
        table.add_column("Retexture", justify="center")
        table.add_column("Error", style="red", max_width=40)

        icons = {
            "SUCCEEDED": "[green]OK[/]",
            "FAILED": "[red]FAIL[/]",
            "IN_PROGRESS": "[yellow]...[/]",
            "PENDING": "[dim]wait[/]",
            "NOT_STARTED": "[dim]--[/]",
            "SKIPPED": "[dim]skip[/]",
        }
        for t in self.tasks:
            table.add_row(
                self._task_name(t),
                icons.get(t.image_to_3d_status, t.image_to_3d_status),
                icons.get(t.remesh_status, t.remesh_status),
                icons.get(t.retexture_status, t.retexture_status),
                t.error or "",
            )

        if self.cfg["image_to_3d_mode"] == "multi":
            succeeded = sum(1 for t in self.tasks if t.remesh_status == "SUCCEEDED")
        else:
            succeeded = sum(1 for t in self.tasks if t.retexture_status == "SUCCEEDED")
        failed = sum(1 for t in self.tasks if t.error)

        console.print()
        console.print(table)
        console.print(
            f"\n[bold green]{succeeded}[/] succeeded, "
            f"[bold red]{failed}[/] failed, "
            f"[bold]{len(self.tasks)}[/] total"
        )

    # -- main entry -----------------------------------------------------------

    def run(self):
        self.discover_images()
        self.step_image_to_3d()
        self.step_remesh()
        self.step_retexture()
        self.download_results()
        self.print_summary()
        console.print(Panel("[bold green]Pipeline complete![/]"))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Meshy AI Bulk Pipeline: Image -> 3D -> Remesh -> Retexture",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
python meshy_pipeline.py -i ./images -o ./results
python meshy_pipeline.py -i ./images -o ./results --polycount 5000
python meshy_pipeline.py -i ./images -o ./results --formats glb fbx obj
python meshy_pipeline.py -i ./images -o ./results --image-to-3d-mode multi
python meshy_pipeline.py -i ./images -o ./results --download-only
        """,
    )

    parser.add_argument("-i", "--input", required=True,
                        help="Folder with source images (.jpg, .jpeg, .png)")
    parser.add_argument("-o", "--output", default="./output",
                        help="Output folder (default: ./output)")
    parser.add_argument("--api-key",
                        help="Meshy API key (or set MESHY_API_KEY env / .env)")

    gen = parser.add_argument_group("Image-to-3D settings")
    gen.add_argument("--ai-model", default="meshy-6",
                     choices=["meshy-5", "meshy-6", "latest"],
                     help="AI model (default: meshy-6)")
    gen.add_argument("--model-type", default="standard",
                     choices=["standard", "lowpoly"],
                     help="Mesh generation type (default: standard)")
    gen.add_argument("--no-image-enhancement", action="store_true",
                     help="Disable image enhancement")
    gen.add_argument("--image-to-3d-mode", default="single",
                     choices=["single", "multi"],
                     help="Image-to-3D endpoint mode (default: single)")
    gen.add_argument("--multi-image-group-size", type=int, default=4,
                     help="Max images per multi-image task (auto-batched), 1-4 (default: 4)")

    rem = parser.add_argument_group("Remesh settings")
    rem.add_argument("--polycount", type=int, default=3000,
                     help="Target polygon count (default: 3000)")
    rem.add_argument("--resize-height", type=float, default=0.05,
                     help="Final model height in meters (default: 0.05)")
    rem.add_argument("--topology", default="triangle",
                     choices=["triangle", "quad"],
                     help="Mesh topology (default: triangle)")
    rem.add_argument("--formats", nargs="+", default=["fbx"],
                     help="Output formats, e.g. glb fbx obj (default: fbx; multi mode forces fbx)")

    tex = parser.add_argument_group("Retexture settings")
    tex.add_argument("--enable-pbr", action="store_true",
                     help="Generate PBR maps (default: off)")
    tex.add_argument("--no-remove-lighting", action="store_true",
                     help="Keep baked lighting (default: removed)")

    run = parser.add_argument_group("Execution")
    run.add_argument("--poll-interval", type=int, default=10,
                     help="Seconds between polls (default: 10)")
    run.add_argument("--submit-delay", type=float, default=1.0,
                     help="Seconds between submissions (default: 1.0)")
    run.add_argument("--download-only", action="store_true",
                     help="Only download results from a previous run")

    args = parser.parse_args()

    if not 1 <= args.multi_image_group_size <= 4:
        parser.error("--multi-image-group-size must be between 1 and 4")

    api_key = args.api_key or os.getenv("MESHY_API_KEY")
    if not api_key:
        console.print(
            "[red]No API key. Use --api-key or set MESHY_API_KEY in "
            "environment / .env file.[/]"
        )
        sys.exit(1)

    target_formats = ["fbx"] if args.image_to_3d_mode == "multi" else args.formats

    cfg = {
        "ai_model": args.ai_model,
        "model_type": args.model_type,
        "image_enhancement": not args.no_image_enhancement,
        "image_to_3d_mode": args.image_to_3d_mode,
        "multi_image_group_size": args.multi_image_group_size,
        "pose_mode": "",
        "target_polycount": args.polycount,
        "resize_height": args.resize_height,
        "topology": args.topology,
        "target_formats": target_formats,
        "retexture_ai_model": args.ai_model,
        "remove_lighting": not args.no_remove_lighting,
        "enable_pbr": args.enable_pbr,
        "enable_original_uv": True,
        "poll_interval": args.poll_interval,
        "submit_delay": args.submit_delay,
    }

    console.print(Panel.fit(
        "[bold]Meshy AI Bulk Pipeline[/]\n"
        f"Input:     {args.input}\n"
        f"Output:    {args.output}\n"
        f"AI Model:  {cfg['ai_model']}  |  Mesh: {cfg['model_type']}\n"
        f"I2M Mode:  {cfg['image_to_3d_mode']}"
        + (f" (group_size={cfg['multi_image_group_size']})" if cfg['image_to_3d_mode'] == 'multi' else "")
        + "\n"
        f"Remesh:    {cfg['target_polycount']} polys ({cfg['topology']}), "
        f"height={cfg['resize_height']}m\n"
        f"Retexture: remove_lighting={cfg['remove_lighting']}, "
        f"pbr={cfg['enable_pbr']}\n"
        f"Formats:   {', '.join(cfg['target_formats'])}",
        title="Config",
    ))

    client = MeshyClient(api_key)
    pipeline = Pipeline(client, args.input, args.output, cfg)

    if args.download_only:
        pipeline.discover_images()
        pipeline.download_results()
        pipeline.print_summary()
    else:
        pipeline.run()


if __name__ == "__main__":
    main()
