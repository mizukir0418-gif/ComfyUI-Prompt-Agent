"""Self-contained WD14 ONNX tiered tagger engine for ComfyUI-Prompt-Agent.

Zero dependency on external custom nodes. Models are stored under
ComfyUI-Prompt-Agent/models/wd14/ (or reused read-only if already present in
ComfyUI/models/wd14_tagger/) and downloaded from HuggingFace on first use
with HTTP Range resume, progress logging, and batch session reuse.
"""

from __future__ import annotations

import asyncio
import csv
import gc
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.request import getproxies

import aiohttp
import numpy as np
from PIL import Image

try:
    import comfy.model_management
except ImportError:
    comfy = None

try:
    import folder_paths
except ImportError:
    folder_paths = None

try:
    import onnxruntime as ort
except ImportError:
    ort = None

logger = logging.getLogger("ComfyUI-Prompt-Agent.WD14")

NODE_DIR = Path(__file__).resolve().parent
LOCAL_WD14_DIR = NODE_DIR / "models" / "wd14"

DEFAULT_WD14_MODEL = "wd-eva02-large-tagger-v3"
WD14_HIGH_CONF_THRESHOLD = 0.70
DEFAULT_HF_ENDPOINT = "https://huggingface.co"

DEFAULT_WD14_EXCLUDE_TAGS = (
    "watermark, web address, artist name, signature, username, twitter username, "
    "patreon username, dated, english text, text, commentary, commentary request, "
    "logo, copyright name, scan, jpeg artifacts, bad id, bad pixiv id, lowres"
)

KNOWN_WD14_MODELS: Dict[str, str] = {
    "wd-eva02-large-tagger-v3": "{HF_ENDPOINT}/SmilingWolf/wd-eva02-large-tagger-v3",
    "wd-vit-tagger-v3": "{HF_ENDPOINT}/SmilingWolf/wd-vit-tagger-v3",
    "wd-swinv2-tagger-v3": "{HF_ENDPOINT}/SmilingWolf/wd-swinv2-tagger-v3",
    "wd-convnext-tagger-v3": "{HF_ENDPOINT}/SmilingWolf/wd-convnext-tagger-v3",
    "wd-v1-4-moat-tagger-v2": "{HF_ENDPOINT}/SmilingWolf/wd-v1-4-moat-tagger-v2",
    "wd-v1-4-convnextv2-tagger-v2": "{HF_ENDPOINT}/SmilingWolf/wd-v1-4-convnextv2-tagger-v2",
    "wd-v1-4-convnext-tagger-v2": "{HF_ENDPOINT}/SmilingWolf/wd-v1-4-convnext-tagger-v2",
    "wd-v1-4-vit-tagger-v2": "{HF_ENDPOINT}/SmilingWolf/wd-v1-4-vit-tagger-v2",
    "wd-v1-4-swinv2-tagger-v2": "{HF_ENDPOINT}/SmilingWolf/wd-v1-4-swinv2-tagger-v2",
}


def _get_proxy_url() -> Optional[str]:
    for key in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
        val = os.environ.get(key)
        if val and val.strip():
            return val.strip()
    try:
        proxies = getproxies()
        for key in ("https", "http"):
            val = proxies.get(key)
            if val and val.strip():
                return val.strip()
    except Exception:
        pass
    return None


def _check_interrupted() -> None:
    if comfy is not None and comfy.model_management.processing_interrupted():
        comfy.model_management.throw_exception_if_processing_interrupted()


def get_wd14_search_dirs() -> List[Path]:
    """Return ordered directories searched for installed WD14 .onnx + .csv pairs."""
    candidates: List[Path] = [LOCAL_WD14_DIR]
    if folder_paths is not None:
        if "wd14_tagger" in folder_paths.folder_names_and_paths:
            for p in folder_paths.get_folder_paths("wd14_tagger"):
                candidates.append(Path(p))
        candidates.append(Path(folder_paths.models_dir) / "wd14_tagger")
    else:
        candidates.append(NODE_DIR.parent.parent / "models" / "wd14_tagger")

    unique: List[Path] = []
    seen: set[str] = set()
    for d in candidates:
        try:
            resolved = d.resolve()
        except OSError:
            continue
        key = str(resolved).lower() if os.name == "nt" else str(resolved)
        if key not in seen:
            seen.add(key)
            unique.append(resolved)
    return unique


def get_wd14_download_dir() -> Path:
    LOCAL_WD14_DIR.mkdir(parents=True, exist_ok=True)
    return LOCAL_WD14_DIR


def discover_installed_wd14_models() -> Dict[str, Tuple[Path, Path]]:
    """Map installed model stem -> (onnx_path, csv_path)."""
    found: Dict[str, Tuple[Path, Path]] = {}
    for search_dir in get_wd14_search_dirs():
        if not search_dir.is_dir():
            continue
        for onnx_file in sorted(search_dir.glob("*.onnx")):
            csv_file = search_dir / f"{onnx_file.stem}.csv"
            if csv_file.is_file():
                found.setdefault(onnx_file.stem, (onnx_file.resolve(), csv_file.resolve()))
    return found


def list_wd14_models() -> List[str]:
    installed = list(discover_installed_wd14_models().keys())
    known = list(KNOWN_WD14_MODELS.keys())
    ordered: List[str] = []
    if DEFAULT_WD14_MODEL in installed or DEFAULT_WD14_MODEL in known:
        ordered.append(DEFAULT_WD14_MODEL)
    for name in installed + known:
        if name not in ordered:
            ordered.append(name)
    return ordered or [DEFAULT_WD14_MODEL]


def _select_ort_providers() -> List[str]:
    if ort is None:
        raise RuntimeError("onnxruntime is not installed. Run: pip install onnxruntime")
    available = set(ort.get_available_providers())
    preferred = [
        "CUDAExecutionProvider",
        "ROCMExecutionProvider",
        "DmlExecutionProvider",
        "CoreMLExecutionProvider",
        "CPUExecutionProvider",
    ]
    selected = [p for p in preferred if p in available]
    return selected if selected else ["CPUExecutionProvider"]


async def _download_file(session: aiohttp.ClientSession, url: str, dest: Path, proxy_url: Optional[str]) -> None:
    """Download file with HTTP Range resume (.part), live progress logging, and cancel support."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest.with_suffix(dest.suffix + ".part")
    existing_bytes = tmp_path.stat().st_size if tmp_path.is_file() else 0

    headers: Dict[str, str] = {"User-Agent": "ComfyUI-Prompt-Agent"}
    if existing_bytes > 0:
        headers["Range"] = f"bytes={existing_bytes}-"
        logger.info("[PromptAgent.WD14] Resuming %s from %.1f MB...", dest.name, existing_bytes / (1024 * 1024))
    else:
        logger.info("[PromptAgent.WD14] Downloading %s ...", dest.name)

    req_kwargs: Dict[str, Any] = {"headers": headers, "allow_redirects": True}
    if proxy_url:
        req_kwargs["proxy"] = proxy_url

    async with session.get(url, **req_kwargs) as resp:
        if resp.status == 416:
            # Range already satisfied
            os.replace(tmp_path, dest)
            return
        if resp.status not in (200, 206):
            raise RuntimeError(
                f"Failed to download {dest.name} (HTTP {resp.status}). "
                "You can set HF_ENDPOINT=https://hf-mirror.com or place the .onnx/.csv files in models/wd14/."
            )

        is_partial = resp.status == 206 and existing_bytes > 0
        mode = "ab" if is_partial else "wb"
        downloaded = existing_bytes if is_partial else 0
        content_len = int(resp.headers.get("Content-Length") or 0)
        total_bytes = (existing_bytes + content_len) if (is_partial and content_len > 0) else content_len

        started_at = time.monotonic()
        last_log = started_at
        session_bytes = 0

        with open(tmp_path, mode) as fh:
            async for chunk in resp.content.iter_chunked(1024 * 256):
                _check_interrupted()
                fh.write(chunk)
                chunk_len = len(chunk)
                downloaded += chunk_len
                session_bytes += chunk_len
                now = time.monotonic()
                if now - last_log >= 2.0:
                    last_log = now
                    elapsed = max(now - started_at, 0.001)
                    speed_mb = (session_bytes / elapsed) / (1024 * 1024)
                    dl_mb = downloaded / (1024 * 1024)
                    if total_bytes > 0:
                        tot_mb = total_bytes / (1024 * 1024)
                        pct = downloaded / total_bytes * 100.0
                        logger.info(
                            "[PromptAgent.WD14] %s: %.1f%% (%.1f / %.1f MB) at %.2f MB/s",
                            dest.name,
                            pct,
                            dl_mb,
                            tot_mb,
                            speed_mb,
                        )
                    else:
                        logger.info("[PromptAgent.WD14] %s: %.1f MB at %.2f MB/s", dest.name, dl_mb, speed_mb)

    os.replace(tmp_path, dest)
    logger.info("[PromptAgent.WD14] Finished downloading %s", dest.name)


async def ensure_wd14_model(model_name: str) -> Tuple[Path, Path]:
    """Resolve an installed WD14 model or download it from HuggingFace into models/wd14/."""
    clean_name = model_name[:-5] if model_name.endswith(".onnx") else model_name
    installed = discover_installed_wd14_models()
    if clean_name in installed:
        return installed[clean_name]

    if clean_name not in KNOWN_WD14_MODELS:
        raise FileNotFoundError(
            f"WD14 model '{clean_name}' is not installed in models/wd14/ and is not in the known model catalog."
        )

    hf_endpoint = os.getenv("HF_ENDPOINT", DEFAULT_HF_ENDPOINT).strip()
    if not hf_endpoint.startswith(("http://", "https://")):
        hf_endpoint = f"https://{hf_endpoint}"
    hf_endpoint = hf_endpoint.rstrip("/")

    repo_url = KNOWN_WD14_MODELS[clean_name].replace("{HF_ENDPOINT}", hf_endpoint)
    base_resolve = f"{repo_url}/resolve/main/"
    download_dir = get_wd14_download_dir()
    onnx_dest = download_dir / f"{clean_name}.onnx"
    csv_dest = download_dir / f"{clean_name}.csv"

    proxy_url = _get_proxy_url()
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=120)
    async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as session:
        if not csv_dest.is_file():
            await _download_file(session, f"{base_resolve}selected_tags.csv", csv_dest, proxy_url)
        if not onnx_dest.is_file():
            await _download_file(session, f"{base_resolve}model.onnx", onnx_dest, proxy_url)

    return onnx_dest.resolve(), csv_dest.resolve()


def prepare_rgb_image(pil_image: Image.Image) -> Image.Image:
    """Composite alpha onto white background and ensure RGB for WD14 & VLM."""
    if pil_image.mode in ("RGBA", "LA") or (pil_image.mode == "P" and "transparency" in pil_image.info):
        rgba = pil_image.convert("RGBA")
        bg = Image.new("RGB", rgba.size, (255, 255, 255))
        bg.paste(rgba, mask=rgba.split()[3])
        return bg
    return pil_image.convert("RGB")


def _escape_danbooru_tag(tag_text: str) -> str:
    return tag_text.replace("(", "\\(").replace(")", "\\)")


def _load_csv_tags(csv_path: Path) -> Tuple[List[str], List[str]]:
    tags: List[str] = []
    categories: List[str] = []
    with open(csv_path, encoding="utf-8", errors="replace") as fh:
        reader = csv.reader(fh)
        next(reader, None)
        for row in reader:
            if len(row) < 3:
                continue
            tags.append(row[1].replace("_", " "))
            categories.append(row[2].strip())
    return tags, categories


def _infer_single_with_session(
    session: Any,
    inp_name: str,
    out_name: str,
    target_height: int,
    tags: List[str],
    categories: List[str],
    rgb_img: Image.Image,
    threshold: float,
    character_threshold: float,
    high_threshold: float,
    excluded: set[str],
) -> Dict[str, Any]:
    ratio = float(target_height) / max(rgb_img.size)
    new_size = (max(1, int(rgb_img.size[0] * ratio)), max(1, int(rgb_img.size[1] * ratio)))
    resized = rgb_img.resize(new_size, Image.LANCZOS)
    square = Image.new("RGB", (target_height, target_height), (255, 255, 255))
    square.paste(resized, ((target_height - new_size[0]) // 2, (target_height - new_size[1]) // 2))

    arr = np.array(square, dtype=np.float32)[:, :, ::-1]  # RGB -> BGR
    arr = np.expand_dims(arr, 0)
    probs = session.run([out_name], {inp_name: arr})[0][0]

    char_pairs: List[Tuple[str, float]] = []
    high_pairs: List[Tuple[str, float]] = []
    med_pairs: List[Tuple[str, float]] = []

    eff_high = max(float(high_threshold), float(threshold))
    for tag_name, cat, prob_raw in zip(tags, categories, probs):
        prob = float(prob_raw)
        if tag_name.lower() in excluded:
            continue
        if cat == "4" and prob > float(character_threshold):
            char_pairs.append((tag_name, prob))
        elif cat == "0" and prob > float(threshold):
            if prob >= eff_high:
                high_pairs.append((tag_name, prob))
            else:
                med_pairs.append((tag_name, prob))

    char_pairs.sort(key=lambda x: x[1], reverse=True)
    high_pairs.sort(key=lambda x: x[1], reverse=True)
    med_pairs.sort(key=lambda x: x[1], reverse=True)

    char_list = [_escape_danbooru_tag(t) for t, _ in char_pairs]
    high_list = [_escape_danbooru_tag(t) for t, _ in high_pairs]
    med_list = [_escape_danbooru_tag(t) for t, _ in med_pairs]
    flat = ", ".join(char_list + high_list + med_list)
    return {
        "character": char_list,
        "high": high_list,
        "medium": med_list,
        "flat": flat,
    }


def _run_onnx_sync(
    rgb_img: Image.Image,
    onnx_path: Path,
    csv_path: Path,
    threshold: float,
    character_threshold: float,
    high_threshold: float,
    exclude_tags: str,
) -> Dict[str, Any]:
    providers = _select_ort_providers()
    tags, categories = _load_csv_tags(csv_path)
    excluded = {s.strip().lower() for s in exclude_tags.split(",") if s.strip()}
    session = ort.InferenceSession(str(onnx_path), providers=providers)
    try:
        inp = session.get_inputs()[0]
        out_name = session.get_outputs()[0].name
        return _infer_single_with_session(
            session=session,
            inp_name=inp.name,
            out_name=out_name,
            target_height=int(inp.shape[1]),
            tags=tags,
            categories=categories,
            rgb_img=rgb_img,
            threshold=threshold,
            character_threshold=character_threshold,
            high_threshold=high_threshold,
            excluded=excluded,
        )
    finally:
        del session


def format_tiered_wd14_item(item: Dict[str, Any]) -> str:
    header = f"[Image {item['index']} ({item['name']})]" if item.get("name") else f"[Image {item['index']}]"
    sublines: List[str] = []
    if item.get("character"):
        sublines.append("  - Character: " + ", ".join(item["character"]))
    if item.get("high"):
        sublines.append(f"  - Core Anchors (High >={WD14_HIGH_CONF_THRESHOLD:.2f}): " + ", ".join(item["high"]))
    if item.get("medium"):
        sublines.append("  - Details (Medium): " + ", ".join(item["medium"]))
    if not sublines:
        return f"{header}: (no tags above threshold)"
    return header + "\n" + "\n".join(sublines)


def _run_batch_groups_sync(
    image_groups: List[List[Tuple[str, Image.Image]]],
    onnx_path: Path,
    csv_path: Path,
    threshold: float,
    character_threshold: float,
    high_threshold: float,
    exclude_tags: str,
    max_images: int,
) -> List[Tuple[List[Dict[str, Any]], str]]:
    """Load ONNX session once, process all image groups in the batch, then immediately free the session."""
    providers = _select_ort_providers()
    tags, categories = _load_csv_tags(csv_path)
    excluded = {s.strip().lower() for s in exclude_tags.split(",") if s.strip()}
    results: List[Tuple[List[Dict[str, Any]], str]] = []

    session = ort.InferenceSession(str(onnx_path), providers=providers)
    try:
        inp = session.get_inputs()[0]
        inp_name = inp.name
        target_height = int(inp.shape[1])
        out_name = session.get_outputs()[0].name

        for group in image_groups:
            _check_interrupted()
            if not group:
                results.append(([], ""))
                continue
            items: List[Dict[str, Any]] = []
            blocks: List[str] = []
            for idx, (img_name, pil_img) in enumerate(group[:max_images], start=1):
                _check_interrupted()
                rgb_img = prepare_rgb_image(pil_img)
                tiered = _infer_single_with_session(
                    session=session,
                    inp_name=inp_name,
                    out_name=out_name,
                    target_height=target_height,
                    tags=tags,
                    categories=categories,
                    rgb_img=rgb_img,
                    threshold=threshold,
                    character_threshold=character_threshold,
                    high_threshold=high_threshold,
                    excluded=excluded,
                )
                item = {
                    "index": idx,
                    "name": img_name,
                    "character": tiered["character"],
                    "high": tiered["high"],
                    "medium": tiered["medium"],
                    "tags": tiered["flat"],
                }
                items.append(item)
                blocks.append(format_tiered_wd14_item(item))
            results.append((items, "\n".join(blocks)))
    finally:
        del session
    return results


async def run_wd14_batch_groups(
    image_groups: List[List[Tuple[str, Image.Image]]],
    model_name: str = DEFAULT_WD14_MODEL,
    threshold: float = 0.35,
    character_threshold: float = 0.85,
    high_threshold: float = WD14_HIGH_CONF_THRESHOLD,
    exclude_tags: str = DEFAULT_WD14_EXCLUDE_TAGS,
    max_images: int = 6,
) -> List[Tuple[List[Dict[str, Any]], str]]:
    """Phase 1 Batch Scheduler: load WD14 ONNX once for all groups, then free VRAM before llama-server starts."""
    if not any(image_groups):
        return [([], "") for _ in image_groups]

    allowed = set(list_wd14_models())
    clean_name = model_name[:-5] if model_name.endswith(".onnx") else model_name
    if clean_name not in allowed:
        clean_name = DEFAULT_WD14_MODEL if DEFAULT_WD14_MODEL in allowed else next(iter(allowed))

    onnx_path, csv_path = await ensure_wd14_model(clean_name)
    batch_out = await asyncio.to_thread(
        _run_batch_groups_sync,
        image_groups,
        onnx_path,
        csv_path,
        float(threshold),
        float(character_threshold),
        float(high_threshold),
        exclude_tags,
        int(max_images),
    )
    gc.collect()
    return batch_out


async def run_wd14_tiered(
    pil_image: Image.Image,
    model_name: str = DEFAULT_WD14_MODEL,
    threshold: float = 0.35,
    character_threshold: float = 0.85,
    high_threshold: float = WD14_HIGH_CONF_THRESHOLD,
    exclude_tags: str = DEFAULT_WD14_EXCLUDE_TAGS,
) -> Dict[str, Any]:
    allowed = set(list_wd14_models())
    clean_name = model_name[:-5] if model_name.endswith(".onnx") else model_name
    if clean_name not in allowed:
        clean_name = DEFAULT_WD14_MODEL if DEFAULT_WD14_MODEL in allowed else next(iter(allowed))

    onnx_path, csv_path = await ensure_wd14_model(clean_name)
    rgb_img = prepare_rgb_image(pil_image)
    result = await asyncio.to_thread(
        _run_onnx_sync,
        rgb_img,
        onnx_path,
        csv_path,
        float(threshold),
        float(character_threshold),
        float(high_threshold),
        exclude_tags,
    )
    gc.collect()
    return result


async def run_wd14_multi(
    pil_images: List[Tuple[str, Image.Image]],
    model_name: str = DEFAULT_WD14_MODEL,
    threshold: float = 0.35,
    character_threshold: float = 0.85,
    exclude_tags: str = DEFAULT_WD14_EXCLUDE_TAGS,
    max_images: int = 6,
) -> Tuple[List[Dict[str, Any]], str]:
    batch_res = await run_wd14_batch_groups(
        [pil_images],
        model_name=model_name,
        threshold=threshold,
        character_threshold=character_threshold,
        exclude_tags=exclude_tags,
        max_images=max_images,
    )
    return batch_res[0]
