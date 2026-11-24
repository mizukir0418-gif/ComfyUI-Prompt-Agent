import asyncio
import atexit
import json
import logging
import os
import platform
import re
import shlex
import shutil
import socket
import subprocess
import tarfile
import tempfile
import threading
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import unquote
from urllib.request import getproxies

import aiohttp

try:
    import folder_paths
except ImportError:
    folder_paths = None

try:
    import torch
except ImportError:
    torch = None

logger = logging.getLogger("ComfyUI-Prompt-Agent")

NODE_DIR = Path(__file__).resolve().parent
COMFY_ROOT = NODE_DIR.parent.parent
BIN_DIR = NODE_DIR / "bin"
CACHE_DIR = NODE_DIR / ".cache"
PROMPT_SKILLS_DIR = NODE_DIR / "prompt_skills"
RELEASE_CACHE_FILE = CACHE_DIR / "llama_cpp_release.json"
KERNEL_SETTINGS_FILE = CACHE_DIR / "kernel_settings.json"
VERSION_META_FILE = BIN_DIR / "version.json"
SERVER_LOG_FILE = CACHE_DIR / "llama-server.log"

MAX_ARCHIVE_FILES = 250
MAX_ARCHIVE_BYTES = 2_500_000_000  # 2.5 GB unpacked limit against Zip Bomb

_FORBIDDEN_EXTRA_FLAGS = {
    "--host",
    "--port",
    "-m",
    "--model",
    "-mm",
    "--mmproj",
    "--path",
    "--log-file",
    "--lora",
    "--lora-scaled",
    "--control-vector",
    "--chat-template-file",
    "--grammar-file",
    "-f",
    "--file",
}


def sanitize_display_path(path_val: Optional[ Any]) -> Optional[str]:
    """Sanitize filesystem paths for UI/API/log output so personal usernames/directories are never leaked."""
    if not path_val:
        return None
    try:
        p = Path(path_val).resolve()
    except OSError:
        return str(path_val)

    try:
        rel_comfy = p.relative_to(COMFY_ROOT.resolve())
        return "<ComfyUI>/" + str(rel_comfy).replace("\\", "/")
    except ValueError:
        pass

    try:
        rel_home = p.relative_to(Path.home().resolve())
        return "~/" + str(rel_home).replace("\\", "/")
    except (ValueError, RuntimeError):
        pass

    return p.name


def resolve_user_path(raw_path: str) -> Path:
    """Resolve user-supplied path supporting `<ComfyUI>/...`, `~/...`, relative, or absolute paths."""
    cleaned = (raw_path or "").strip().strip("\"'")
    if cleaned.startswith("<ComfyUI>/") or cleaned.startswith("<ComfyUI>\\"):
        return (COMFY_ROOT / cleaned[len("<ComfyUI>/") :]).resolve()
    p = Path(cleaned).expanduser()
    if not p.is_absolute():
        candidate_comfy = (COMFY_ROOT / p).resolve()
        if candidate_comfy.exists():
            return candidate_comfy
        candidate_node = (NODE_DIR / p).resolve()
        if candidate_node.exists():
            return candidate_node
    return p.resolve()

GITHUB_REPO = "ggml-org/llama.cpp"
GITHUB_LATEST_RELEASE_API = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
GITHUB_LATEST_RELEASE_WEB = f"https://github.com/{GITHUB_REPO}/releases/latest"
GITHUB_RELEASES_WEB = f"https://github.com/{GITHUB_REPO}/releases"
GITHUB_TAG_WEB_BASE = f"https://github.com/{GITHUB_REPO}/releases/tag"
GITHUB_DOWNLOAD_BASE = f"https://github.com/{GITHUB_REPO}/releases/download"
GITHUB_EXPANDED_ASSETS_BASE = f"https://github.com/{GITHUB_REPO}/releases/expanded_assets"

RELEASE_CACHE_TTL_SECONDS = 24 * 3600  # 24 hours to avoid GitHub API rate limits
MAX_LLAMA_SEED = (2**32) - 1

_SEMVER_RE = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)(?:[-+].*)?$", re.IGNORECASE)
_BUILD_RE = re.compile(r"^b?(\d+)$", re.IGNORECASE)
_VERSION_OUTPUT_SEMVER_RE = re.compile(r"\bv?(\d+\.\d+\.\d+)\b")
_VERSION_OUTPUT_BUILD_RE = re.compile(r"version:\s*(\d+)\b", re.IGNORECASE)
_VERSION_OUTPUT_BTAG_RE = re.compile(r"\bb(\d{3,6})\b", re.IGNORECASE)
_MMPROJ_MISMATCH_RE = re.compile(
    r"mismatch between text model \(n_embd = (?P<model>\d+)\) and mmproj \(n_embd = (?P<mmproj>\d+)\)",
    flags=re.IGNORECASE,
)
_EXPANDED_HREF_RE = re.compile(r'href="(/ggml-org/llama\.cpp/releases/download/[^"]+)"')


def _is_windows() -> bool:
    return os.name == "nt"


def _executable_name() -> str:
    return "llama-server.exe" if _is_windows() else "llama-server"


def _normalize_version_tag(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    val = raw.strip()
    if not val:
        return None
    m_sem = _SEMVER_RE.match(val)
    if m_sem:
        return f"v{m_sem.group(1)}.{m_sem.group(2)}.{m_sem.group(3)}"
    m_build = _BUILD_RE.match(val)
    if m_build:
        return f"b{int(m_build.group(1))}"
    return val


def _parse_version_key(tag: Optional[str]) -> Tuple[str, Tuple[int, ...]]:
    norm = _normalize_version_tag(tag)
    if not norm:
        return ("none", ())
    m_sem = _SEMVER_RE.match(norm)
    if m_sem:
        return ("semver", (int(m_sem.group(1)), int(m_sem.group(2)), int(m_sem.group(3))))
    m_build = _BUILD_RE.match(norm)
    if m_build:
        return ("build", (int(m_build.group(1)),))
    return ("raw", ())


def is_newer_version(
    local_ver: Optional[str],
    latest_ver: Optional[str],
    latest_binary_tag: Optional[str] = None,
) -> bool:
    """Compare local llama.cpp version against latest release tag (and optional resolved build tag)."""
    if not latest_ver and not latest_binary_tag:
        return False
    if not local_ver:
        return True

    local_norm = _normalize_version_tag(local_ver)
    latest_norm = _normalize_version_tag(latest_ver)
    binary_norm = _normalize_version_tag(latest_binary_tag)

    # Already matching either semantic release tag or resolved nightly build tag
    if local_norm and local_norm in {latest_norm, binary_norm}:
        return False

    local_kind, local_tuple = _parse_version_key(local_norm)

    # Prefer same-scheme comparison when available (e.g. local b10472 vs resolved b12359)
    for candidate in (latest_norm, binary_norm):
        if not candidate:
            continue
        cand_kind, cand_tuple = _parse_version_key(candidate)
        if local_kind == cand_kind and local_kind in ("semver", "build"):
            return cand_tuple > local_tuple

    return local_norm != (latest_norm or binary_norm)


def _parse_version_from_cli_output(text: str) -> Optional[str]:
    if not text:
        return None
    # Inspect the dedicated `version:` line first so compiler versions (e.g. Clang 20.1.8) never match
    version_line = ""
    for line in text.splitlines():
        if "version:" in line.lower():
            version_line = line.strip()
            break
    target = version_line or text

    # Pre-release / nightly format: `version: 0.1.1-dev (build 10472, commit ...)` -> b10472
    m_dev_build = re.search(r"-dev\b.*?\bbuild\s+(\d+)", target, re.IGNORECASE)
    if m_dev_build:
        return f"b{int(m_dev_build.group(1))}"

    # Stable semver format: `version: 1.0.0 (build 12359, ...)` -> v1.0.0
    m_sem = re.search(r"version:\s*v?(\d+\.\d+\.\d+)(?!-dev)", target, re.IGNORECASE)
    if m_sem:
        return f"v{m_sem.group(1)}"

    # Generic build formats: `(build 10472)` / `b10472` / `version: 5123`
    m_build_word = re.search(r"\bbuild\s+(\d+)\b", target, re.IGNORECASE)
    if m_build_word:
        return f"b{int(m_build_word.group(1))}"
    m_btag = _VERSION_OUTPUT_BTAG_RE.search(target)
    if m_btag:
        return f"b{int(m_btag.group(1))}"
    m_build = _VERSION_OUTPUT_BUILD_RE.search(target)
    if m_build:
        build_num = int(m_build.group(1))
        if build_num > 0:
            return f"b{build_num}"
    return None


def split_extra_args(extra_args: str) -> List[str]:
    if not extra_args or not extra_args.strip():
        return []
    parts = shlex.split(extra_args, posix=(not _is_windows()))
    cleaned: List[str] = []
    for part in parts:
        token = part.strip("\"'")
        if not token:
            continue
        flag_name = token.split("=", 1)[0].lower()
        if flag_name in _FORBIDDEN_EXTRA_FLAGS:
            raise ValueError(f"Disallowed flag in extra_args: '{flag_name}'")
        cleaned.append(token)
    return cleaned


def normalize_llama_seed(seed: int) -> int:
    seed = int(seed)
    if seed < 0:
        return -1
    if seed <= MAX_LLAMA_SEED:
        return seed
    return seed % (MAX_LLAMA_SEED + 1)


def _pick_free_port(preferred: int = 8088) -> int:
    for port in (preferred, 0):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
                return sock.getsockname()[1]
            except OSError:
                continue
    raise RuntimeError("Unable to bind a local TCP port for llama-server.")


_GPU_INFO_CACHE: Optional[Dict[str, Any]] = None


def _classify_nvidia_arch(gpu_name: str, cc: Optional[Tuple[int, int]] = None) -> Tuple[str, bool]:
    """Return (architecture_family, is_legacy_pre_turing) for NVIDIA GPUs."""
    name_up = (gpu_name or "").upper()
    if cc:
        major, minor = cc
        if major >= 10:
            return "Blackwell", False
        if (major, minor) == (9, 0):
            return "Hopper", False
        if (major, minor) == (8, 9):
            return "Ada Lovelace", False
        if major == 8:
            return "Ampere", False
        if (major, minor) == (7, 5):
            return "Turing", False
        if major == 7:
            return "Volta", True
        if major == 6:
            return "Pascal", True
        if major <= 5:
            return "Maxwell/Older", True

    if re.search(r"\b(RTX\s*50\d{2}|RTX\s*PRO\s*\d+|B100|B200|GB200)\b", name_up):
        return "Blackwell", False
    if re.search(r"\b(RTX\s*40\d{2}|L40|L4|RTX\s*\d+\s*ADA)\b", name_up):
        return "Ada Lovelace", False
    if re.search(r"\b(RTX\s*30\d{2}|A100|A6000|A5000|A4000|A10|A40)\b", name_up):
        return "Ampere", False
    if re.search(r"\b(RTX\s*20\d{2}|GTX\s*16\d{2}|T4|QUADRO\s*RTX)\b", name_up):
        return "Turing", False
    if re.search(r"\b(GTX\s*10\d{2}|P100|P40|TITAN\s*X[P]?)\b", name_up):
        return "Pascal", True
    if re.search(r"\b(GTX\s*9\d{2}|GTX\s*7\d{2})\b", name_up):
        return "Maxwell/Older", True
    return "NVIDIA", False


def _probe_nvidia_smi() -> Optional[Dict[str, Any]]:
    smi = shutil.which("nvidia-smi")
    if not smi:
        return None
    kwargs: Dict[str, Any] = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
        "timeout": 4,
    }
    if _is_windows():
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

    gpu_name = ""
    driver_ver = ""
    driver_cuda: Optional[Tuple[int, int]] = None
    try:
        q = subprocess.run(
            [smi, "--query-gpu=name,driver_version", "--format=csv,noheader"],
            **kwargs,
        )
        if q.returncode == 0 and q.stdout.strip():
            first_line = q.stdout.strip().splitlines()[0]
            parts = [p.strip() for p in first_line.split(",")]
            if parts:
                gpu_name = parts[0]
            if len(parts) > 1:
                driver_ver = parts[1]
    except Exception:
        pass

    try:
        full = subprocess.run([smi], **kwargs)
        header = (full.stdout or "") + "\n" + (full.stderr or "")
        # Matches both `CUDA Version: 12.8` and `CUDA UMD Version: 13.5`
        m = re.search(r"CUDA(?:\s+UMD)?\s+Version:\s*(\d+)\.(\d+)", header, re.IGNORECASE)
        if m:
            driver_cuda = (int(m.group(1)), int(m.group(2)))
    except Exception:
        pass

    if gpu_name or driver_cuda:
        return {
            "gpu_name": gpu_name,
            "driver_version": driver_ver,
            "driver_cuda": driver_cuda,
        }
    return None


def _probe_windows_gpu_names() -> List[str]:
    if not _is_windows():
        return []
    try:
        proc = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "(Get-CimInstance Win32_VideoController).Name",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=4,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        if proc.returncode == 0 and proc.stdout:
            return [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    except Exception:
        pass
    return []


def detect_gpu_hardware(refresh: bool = False) -> Dict[str, Any]:
    """Detect actual GPU vendor, model, architecture, and driver CUDA version."""
    global _GPU_INFO_CACHE
    if _GPU_INFO_CACHE is not None and not refresh:
        return dict(_GPU_INFO_CACHE)

    torch_cuda: Optional[Tuple[int, int]] = None
    torch_hip: Optional[str] = None
    torch_gpu_name = ""
    torch_cc: Optional[Tuple[int, int]] = None

    if torch is not None:
        if getattr(torch.version, "cuda", None):
            m = re.match(r"(\d+)\.(\d+)", str(torch.version.cuda))
            if m:
                torch_cuda = (int(m.group(1)), int(m.group(2)))
        if getattr(torch.version, "hip", None):
            torch_hip = str(torch.version.hip)
        try:
            if torch.cuda.is_available():
                torch_gpu_name = str(torch.cuda.get_device_name(0) or "")
                cc = torch.cuda.get_device_capability(0)
                if isinstance(cc, tuple) and len(cc) == 2:
                    torch_cc = (int(cc[0]), int(cc[1]))
        except Exception:
            pass

    vendor = "cpu"
    gpu_name = ""
    arch_family = ""
    driver_version = ""
    driver_cuda: Optional[Tuple[int, int]] = None
    is_legacy_nvidia = False

    # 1. Check macOS Apple Silicon / Metal
    if platform.system().lower() == "darwin":
        vendor = "apple"
        gpu_name = "Apple Silicon (Metal)" if platform.machine().lower() in ("arm64", "aarch64") else "macOS GPU"
        arch_family = "Metal"
    else:
        # 2. Check NVIDIA via nvidia-smi (gives true driver CUDA version, independent of PyTorch!)
        nv = _probe_nvidia_smi()
        if nv or (torch_cuda and not torch_hip):
            vendor = "nvidia"
            gpu_name = (nv.get("gpu_name") if nv else "") or torch_gpu_name or "NVIDIA GPU"
            driver_version = (nv.get("driver_version") if nv else "") or ""
            driver_cuda = (nv.get("driver_cuda") if nv else None) or torch_cuda
            arch_family, is_legacy_nvidia = _classify_nvidia_arch(gpu_name, torch_cc)
        elif torch_hip:
            vendor = "amd"
            gpu_name = torch_gpu_name or "AMD Radeon GPU"
            arch_family = f"ROCm/HIP {torch_hip}"
        else:
            # 3. Inspect OS video controllers (detects AMD Radeon / Intel Arc when PyTorch is CPU-only)
            win_gpus = _probe_windows_gpu_names()
            for name in win_gpus:
                low = name.lower()
                if "nvidia" in low or "geforce" in low or "rtx" in low or "gtx" in low:
                    vendor = "nvidia"
                    gpu_name = name
                    arch_family, is_legacy_nvidia = _classify_nvidia_arch(gpu_name, torch_cc)
                    break
            if vendor == "cpu":
                for name in win_gpus:
                    low = name.lower()
                    if "amd" in low or "radeon" in low:
                        vendor = "amd"
                        gpu_name = name
                        arch_family = "Radeon"
                        break
            if vendor == "cpu":
                for name in win_gpus:
                    low = name.lower()
                    if "intel" in low and ("arc" in low or "iris" in low or "uhd" in low or "graphics" in low):
                        vendor = "intel"
                        gpu_name = name
                        arch_family = "Intel Arc" if "arc" in low else "Intel Graphics"
                        break

    details: List[str] = []
    if arch_family:
        details.append(arch_family)
    if driver_version:
        details.append(f"Driver {driver_version}")
    if driver_cuda:
        details.append(f"Driver CUDA {driver_cuda[0]}.{driver_cuda[1]}")
    elif torch_cuda:
        details.append(f"PyTorch CUDA {torch_cuda[0]}.{torch_cuda[1]}")

    summary = f"{gpu_name} ({' · '.join(details)})" if gpu_name and details else (gpu_name or "CPU only (no discrete GPU detected)")

    _GPU_INFO_CACHE = {
        "vendor": vendor,
        "gpu_name": gpu_name,
        "arch_family": arch_family,
        "compute_capability": list(torch_cc) if torch_cc else None,
        "driver_version": driver_version,
        "driver_cuda": list(driver_cuda) if driver_cuda else None,
        "torch_cuda": list(torch_cuda) if torch_cuda else None,
        "torch_hip": torch_hip,
        "is_legacy_nvidia": is_legacy_nvidia,
        "summary": summary,
    }
    return dict(_GPU_INFO_CACHE)


def _detect_platform_info() -> Dict[str, Any]:
    sys_name = platform.system().lower()
    machine = platform.machine().lower()
    if machine in ("amd64", "x86_64", "x64"):
        arch = "x64"
    elif machine in ("arm64", "aarch64"):
        arch = "arm64"
    else:
        arch = machine

    gpu = detect_gpu_hardware()
    driver_cuda = tuple(gpu["driver_cuda"]) if gpu.get("driver_cuda") else None

    if sys_name == "windows":
        os_key = "win"
    elif sys_name == "darwin":
        os_key = "macos"
    else:
        os_key = "linux"

    return {
        "os": os_key,
        "arch": arch,
        "cuda_version": driver_cuda,
        "gpu": gpu,
        "recommended_backend": "metal" if os_key == "macos" else ("cuda" if gpu["vendor"] == "nvidia" else "vulkan"),
    }


def _extract_cuda_tuple_from_name(name: str) -> Optional[Tuple[int, int]]:
    lower = name.lower()
    m = re.search(r"(?:cuda[-_]?|cu)(\d+)\.(\d+)", lower)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    m2 = re.search(r"cu(\d{2})(\d)", lower)
    if m2:
        return (int(m2.group(1)), int(m2.group(2)))
    return None


def _has_platform_binary_archives(assets: List[Dict[str, Any]]) -> bool:
    for a in assets:
        name = (a.get("name") or "").lower()
        if ("bin-win-" in name or "bin-ubuntu-" in name or "bin-macos-" in name) and (
            name.endswith(".zip") or name.endswith(".tar.gz")
        ):
            return True
    return False


def match_release_assets(assets: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Match ggml-org/llama.cpp release archives and recommend the optimal backend for the user's GPU."""
    plat = _detect_platform_info()
    os_key = plat["os"]
    arch = plat["arch"]
    gpu = plat["gpu"]
    vendor = gpu["vendor"]
    driver_cuda: Optional[Tuple[int, int]] = plat["cuda_version"]
    is_legacy_nvidia = bool(gpu.get("is_legacy_nvidia"))

    matched: Dict[str, Dict[str, Any]] = {}
    recommended: Optional[str] = None
    reason = ""

    if os_key == "win":
        win_bins = [
            a for a in assets
            if a.get("name", "").lower().endswith(".zip")
            and "bin-win-" in a.get("name", "").lower()
            and not a.get("name", "").lower().startswith("cudart-")
            and (arch in a.get("name", "").lower() or (arch == "x64" and "arm64" not in a.get("name", "").lower()))
        ]
        cudarts = [
            a for a in assets
            if a.get("name", "").lower().endswith(".zip")
            and a.get("name", "").lower().startswith("cudart-")
            and "win" in a.get("name", "").lower()
            and (arch in a.get("name", "").lower() or (arch == "x64" and "arm64" not in a.get("name", "").lower()))
        ]

        # 1. All NVIDIA CUDA builds (sorted descending: e.g. cuda-13.5, cuda-12.4)
        cuda_bins = [a for a in win_bins if "cuda" in a["name"].lower()]
        cuda_bins.sort(key=lambda a: _extract_cuda_tuple_from_name(a["name"]) or (0, 0), reverse=True)
        cuda_entries: List[Tuple[str, Optional[Tuple[int, int]]]] = []

        for asset in cuda_bins:
            ver = _extract_cuda_tuple_from_name(asset["name"])
            key = f"cuda-{ver[0]}.{ver[1]}" if ver else "cuda"
            if key in matched:
                continue
            extra_assets = []
            if ver:
                for rt in cudarts:
                    if _extract_cuda_tuple_from_name(rt["name"]) == ver:
                        extra_assets.append(rt)
                        break
            matched[key] = {
                "primary": asset,
                "extras": extra_assets,
                "label": f"NVIDIA CUDA {ver[0]}.{ver[1]}" if ver else "NVIDIA CUDA",
            }
            cuda_entries.append((key, ver))

        # 2. AMD ROCm / HIP build (if available in release)
        rocm_bins = [a for a in win_bins if ("rocm" in a["name"].lower() or "hip" in a["name"].lower())]
        if rocm_bins:
            rocm_bins.sort(key=lambda a: a["name"], reverse=True)
            matched["rocm"] = {"primary": rocm_bins[0], "extras": [], "label": "AMD ROCm / HIP"}

        # 3. Intel SYCL build (if available in release)
        sycl_bins = [a for a in win_bins if "sycl" in a["name"].lower()]
        if sycl_bins:
            matched["sycl"] = {"primary": sycl_bins[0], "extras": [], "label": "Intel SYCL"}

        # 4. Universal Vulkan build
        vulkan_bins = [a for a in win_bins if "vulkan" in a["name"].lower()]
        if vulkan_bins:
            matched["vulkan"] = {"primary": vulkan_bins[0], "extras": [], "label": "Vulkan (Universal GPU)"}

        # 5. CPU build
        cpu_bins = [
            a for a in win_bins
            if ("cpu" in a["name"].lower() or "avx2" in a["name"].lower())
            and "cuda" not in a["name"].lower()
            and "vulkan" not in a["name"].lower()
            and "sycl" not in a["name"].lower()
            and "hip" not in a["name"].lower()
            and "rocm" not in a["name"].lower()
            and "opencl" not in a["name"].lower()
            and "openvino" not in a["name"].lower()
        ]
        if cpu_bins:
            cpu_bins.sort(key=lambda a: (0 if "avx2" in a["name"].lower() else 1, a["name"]))
            matched["cpu"] = {"primary": cpu_bins[0], "extras": [], "label": "CPU Only"}

        # GPU-model-aware recommendation
        if vendor == "nvidia" and cuda_entries:
            if is_legacy_nvidia:
                # GTX 10xx/9xx (Pascal/Maxwell): CUDA 13.x dropped sm_61/sm_52; prefer CUDA 12.x <= driver_cuda or Vulkan
                c12 = [k for k, v in cuda_entries if v and v[0] == 12 and (not driver_cuda or v <= driver_cuda)]
                if c12:
                    recommended = c12[0]
                    reason = f"{gpu['gpu_name']} ({gpu['arch_family']}) requires CUDA 12.x (CUDA 13 dropped Pascal/Maxwell support)."
                elif "vulkan" in matched:
                    recommended = "vulkan"
                    reason = f"{gpu['gpu_name']} ({gpu['arch_family']}) is best supported via Vulkan on newer llama.cpp builds."
            else:
                compat = [k for k, v in cuda_entries if v and driver_cuda and v <= driver_cuda]
                if compat:
                    recommended = compat[0]
                    reason = f"Matched {gpu['gpu_name']} ({gpu['arch_family']}) with driver CUDA {driver_cuda[0]}.{driver_cuda[1]}."
                else:
                    recommended = cuda_entries[-1][0]
                    reason = f"Selected compatible CUDA build for {gpu['gpu_name']}."
        elif vendor == "amd":
            if gpu.get("torch_hip") and "rocm" in matched:
                recommended = "rocm"
                reason = f"Detected {gpu['gpu_name']} with active ROCm/HIP runtime."
            elif "vulkan" in matched:
                recommended = "vulkan"
                reason = f"Recommended zero-SDK Vulkan backend for {gpu['gpu_name']} (ROCm also selectable if installed)."
            elif "rocm" in matched:
                recommended = "rocm"
                reason = f"Matched AMD ROCm build for {gpu['gpu_name']}."
        elif vendor == "intel":
            if "vulkan" in matched:
                recommended = "vulkan"
                reason = f"Recommended zero-dependency Vulkan backend for {gpu['gpu_name']} (Intel SYCL also selectable)."
            elif "sycl" in matched:
                recommended = "sycl"
                reason = f"Matched Intel SYCL build for {gpu['gpu_name']}."
        else:
            if "cpu" in matched:
                recommended = "cpu"
                reason = "No discrete GPU detected; recommended CPU build."
            elif "vulkan" in matched:
                recommended = "vulkan"
                reason = "Selected universal Vulkan build."

    elif os_key == "macos":
        mac_bins = [
            a for a in assets
            if (a.get("name", "").lower().endswith(".zip") or a.get("name", "").lower().endswith(".tar.gz"))
            and "macos" in a.get("name", "").lower()
            and arch in a.get("name", "").lower()
        ]
        if mac_bins:
            matched["metal"] = {"primary": mac_bins[0], "extras": [], "label": "macOS Metal"}
            recommended = "metal"
            reason = f"Native macOS Metal acceleration ({arch})."

    else:
        linux_bins = [
            a for a in assets
            if (a.get("name", "").lower().endswith(".zip") or a.get("name", "").lower().endswith(".tar.gz"))
            and ("ubuntu" in a.get("name", "").lower() or "linux" in a.get("name", "").lower())
            and (arch in a.get("name", "").lower() or (arch == "x64" and "arm64" not in a.get("name", "").lower()))
        ]
        cuda_bins = [a for a in linux_bins if "cuda" in a["name"].lower()]
        if cuda_bins:
            matched["cuda"] = {"primary": cuda_bins[0], "extras": [], "label": "NVIDIA CUDA"}
        rocm_bins = [a for a in linux_bins if "rocm" in a["name"].lower() or "hip" in a["name"].lower()]
        if rocm_bins:
            matched["rocm"] = {"primary": rocm_bins[0], "extras": [], "label": "AMD ROCm"}
        vulkan_bins = [a for a in linux_bins if "vulkan" in a["name"].lower()]
        if vulkan_bins:
            matched["vulkan"] = {"primary": vulkan_bins[0], "extras": [], "label": "Vulkan (Universal GPU)"}
        cpu_bins = [
            a for a in linux_bins
            if "vulkan" not in a["name"].lower()
            and "cuda" not in a["name"].lower()
            and "rocm" not in a["name"].lower()
            and "sycl" not in a["name"].lower()
        ]
        if cpu_bins:
            matched["cpu"] = {"primary": cpu_bins[0], "extras": [], "label": "CPU Only"}

        if vendor == "nvidia" and "cuda" in matched and not is_legacy_nvidia:
            recommended = "cuda"
            reason = f"Matched NVIDIA CUDA build for {gpu['gpu_name']}."
        elif vendor == "amd" and "rocm" in matched:
            recommended = "rocm"
            reason = f"Matched AMD ROCm build for {gpu['gpu_name']}."
        elif "vulkan" in matched:
            recommended = "vulkan"
            reason = f"Recommended Vulkan build for {gpu['gpu_name'] or 'Linux GPU'}."
        elif "cpu" in matched:
            recommended = "cpu"
            reason = "Selected CPU build."

    if (not recommended or recommended not in matched) and matched:
        recommended = next(iter(matched.keys()))

    return {
        "platform": plat,
        "hardware": gpu,
        "recommended_backend": recommended or plat["recommended_backend"],
        "recommendation_reason": reason,
        "available_backends": list(matched.keys()),
        "matched": matched,
    }


def _safe_extract_archive(archive_path: Path, dest_dir: Path) -> None:
    """Safely extract .zip or .tar.gz archive, blocking Zip Slip path traversal and Zip Bombs."""
    dest_resolved = dest_dir.resolve()
    name_lower = archive_path.name.lower()
    extracted_files = 0
    extracted_bytes = 0

    def _copy_limited(src_obj, dst_obj) -> None:
        nonlocal extracted_bytes
        while True:
            chunk = src_obj.read(1024 * 256)
            if not chunk:
                break
            extracted_bytes += len(chunk)
            if extracted_bytes > MAX_ARCHIVE_BYTES:
                raise ValueError("Archive exceeds maximum allowed uncompressed size (2.5 GB).")
            dst_obj.write(chunk)

    if name_lower.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zf:
            for member in zf.infolist():
                if member.is_dir():
                    continue
                extracted_files += 1
                if extracted_files > MAX_ARCHIVE_FILES:
                    raise ValueError(f"Archive contains too many files (>{MAX_ARCHIVE_FILES}).")
                target = (dest_resolved / member.filename).resolve()
                if not str(target).startswith(str(dest_resolved) + os.sep) and target != dest_resolved:
                    raise ValueError(f"Unsafe zip member path: {member.filename}")
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member, "r") as src, open(target, "wb") as dst:
                    _copy_limited(src, dst)
    elif name_lower.endswith(".tar.gz") or name_lower.endswith(".tgz"):
        with tarfile.open(archive_path, "r:gz") as tf:
            for member in tf.getmembers():
                if not member.isfile():
                    continue
                extracted_files += 1
                if extracted_files > MAX_ARCHIVE_FILES:
                    raise ValueError(f"Archive contains too many files (>{MAX_ARCHIVE_FILES}).")
                target = (dest_resolved / member.name).resolve()
                if not str(target).startswith(str(dest_resolved) + os.sep) and target != dest_resolved:
                    raise ValueError(f"Unsafe tar member path: {member.name}")
                target.parent.mkdir(parents=True, exist_ok=True)
                extracted = tf.extractfile(member)
                if extracted is not None:
                    with extracted, open(target, "wb") as dst:
                        _copy_limited(extracted, dst)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path.name}")


def _flatten_extracted_binaries(staging_dir: Path, target_bin_dir: Path) -> Path:
    """Move extracted runtime binaries and shared libraries into target_bin_dir."""
    exe_name = _executable_name()
    found_exe: Optional[Path] = None
    for candidate in staging_dir.rglob(exe_name):
        if candidate.is_file():
            found_exe = candidate
            break

    if found_exe is None:
        raise FileNotFoundError(f"Archive did not contain {exe_name}")

    source_root = found_exe.parent
    target_bin_dir.mkdir(parents=True, exist_ok=True)

    for item in source_root.iterdir():
        if item.is_file():
            dst = target_bin_dir / item.name
            shutil.copy2(item, dst)
            if not _is_windows() and item.name == exe_name:
                os.chmod(dst, 0o755)

    for lib_ext in ("*.dll", "*.so*", "*.dylib"):
        for lib_file in staging_dir.rglob(lib_ext):
            if lib_file.is_file():
                dst = target_bin_dir / lib_file.name
                if not dst.exists() or lib_file.parent != source_root:
                    shutil.copy2(lib_file, dst)

    return target_bin_dir / exe_name


def _parse_llama_error(log_tail: str) -> str:
    m = _MMPROJ_MISMATCH_RE.search(log_tail or "")
    if m:
        return (
            "Selected mmproj does not match the text model "
            f"(model n_embd={m.group('model')}, mmproj n_embd={m.group('mmproj')}). "
            "Choose the mmproj file that belongs to the selected GGUF model."
        )
    return ""


def _format_timings(timings: Optional[Dict[str, Any]]) -> str:
    if not isinstance(timings, dict):
        return ""
    parts: List[str] = []
    p_tps = timings.get("prompt_per_second")
    p_n = timings.get("prompt_n")
    g_tps = timings.get("predicted_per_second")
    g_n = timings.get("predicted_n")
    if p_tps is not None:
        p_str = f"Prompt: {float(p_tps):.1f} t/s"
        if p_n is not None:
            p_str += f" ({int(p_n)} tok)"
        parts.append(p_str)
    if g_tps is not None:
        g_str = f"Generation: {float(g_tps):.1f} t/s"
        if g_n is not None:
            g_str += f" ({int(g_n)} tok)"
        parts.append(g_str)
    return f"[{' | '.join(parts)}]" if parts else ""


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


class LlamaServerManager:
    """Manages decoupled llama-server lifecycle, GGUF/prompt discovery, and GitHub update checks."""

    def __init__(self) -> None:
        self._proc_lock = threading.RLock()
        self._lifecycle_lock = threading.Lock()
        self._process: Optional[subprocess.Popen] = None
        self._log_fh = None
        self._port: Optional[int] = None
        self._current_config: Optional[Tuple[Any, ...]] = None
        self.auto_unload_on_queue: bool = True

        self._update_task: Optional[asyncio.Task] = None
        self._update_job: Dict[str, Any] = {
            "state": "idle",
            "stage": "",
            "backend": "",
            "current_file": "",
            "file_index": 0,
            "file_count": 0,
            "downloaded_bytes": 0,
            "total_bytes": 0,
            "percent": 0.0,
            "speed_bps": 0.0,
            "error": None,
            "result": None,
        }

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        atexit.register(self.unload)

    # ------------------------------------------------------------------
    # Dynamic Model, Projector & System Prompt Discovery
    # ------------------------------------------------------------------

    def get_model_search_dirs(self) -> List[Path]:
        candidates: List[Path] = []
        if folder_paths is not None:
            for key in ("LLM", "llm", "llms", "gguf", "GGUF"):
                if key in folder_paths.folder_names_and_paths:
                    for p in folder_paths.get_folder_paths(key):
                        candidates.append(Path(p))
            models_root = Path(folder_paths.models_dir)
        else:
            models_root = NODE_DIR.parent.parent / "models"

        for sub in ("LLM", "llm", "llms", "GGUF", "gguf"):
            candidates.append(models_root / sub)

        extra_env = os.environ.get("COMFYUI_LLM_DIR")
        if extra_env:
            candidates.append(Path(extra_env))

        unique_dirs: List[Path] = []
        seen: set[str] = set()
        for d in candidates:
            try:
                resolved = d.resolve()
            except OSError:
                continue
            norm_key = str(resolved).lower() if _is_windows() else str(resolved)
            if norm_key not in seen and resolved.is_dir():
                seen.add(norm_key)
                unique_dirs.append(resolved)
        return unique_dirs

    def get_prompt_search_dirs(self) -> List[Path]:
        candidates: List[Path] = [NODE_DIR / "prompts"]
        for model_dir in self.get_model_search_dirs():
            candidates.append(model_dir / "prompts")

        unique_dirs: List[Path] = []
        seen: set[str] = set()
        for d in candidates:
            try:
                resolved = d.resolve()
            except OSError:
                continue
            norm_key = str(resolved).lower() if _is_windows() else str(resolved)
            if norm_key not in seen and resolved.is_dir():
                seen.add(norm_key)
                unique_dirs.append(resolved)
        return unique_dirs

    def discover_gguf_files(self) -> Tuple[Dict[str, Path], Dict[str, Path]]:
        llm_map: Dict[str, Path] = {}
        mmproj_map: Dict[str, Path] = {}

        for base_dir in self.get_model_search_dirs():
            for root, dirs, files in os.walk(base_dir):
                rel_depth = len(Path(root).relative_to(base_dir).parts)
                if rel_depth >= 2:
                    dirs[:] = []
                for fname in sorted(files):
                    if not fname.lower().endswith(".gguf"):
                        continue
                    full_path = (Path(root) / fname).resolve()
                    rel_name = str(full_path.relative_to(base_dir)).replace("\\", "/")
                    if "mmproj" in fname.lower():
                        mmproj_map.setdefault(rel_name, full_path)
                    else:
                        llm_map.setdefault(rel_name, full_path)

        return llm_map, mmproj_map

    def discover_system_prompts(self) -> Dict[str, Path]:
        prompt_map: Dict[str, Path] = {}
        for pdir in self.get_prompt_search_dirs():
            for txt_file in sorted(pdir.glob("*.txt")):
                if txt_file.is_file():
                    prompt_map.setdefault(txt_file.name, txt_file.resolve())
        return prompt_map

    def discover_prompt_skills(self) -> Dict[str, Path]:
        """Discover optional user-added runtime prompt skill files (.md / .txt) in prompt_skills/."""
        skill_map: Dict[str, Path] = {}
        if PROMPT_SKILLS_DIR.is_dir():
            for ext in ("*.md", "*.txt"):
                for f in sorted(PROMPT_SKILLS_DIR.glob(ext)):
                    if f.is_file():
                        skill_map[f"skill:{f.name}"] = f.resolve()
        return skill_map

    def list_output_styles(self) -> List[str]:
        builtin = [
            "auto",
            "natural_language",
            "danbooru_tags",
            "hybrid",
            "edit_instruction",
            "structured_slots",
            "custom_preset",
        ]
        return builtin + sorted(self.discover_prompt_skills().keys())

    def load_prompt_skill_text(self, style_name: str) -> Optional[str]:
        if not style_name or not style_name.startswith("skill:"):
            return None
        skill_map = self.discover_prompt_skills()
        if style_name not in skill_map:
            raise FileNotFoundError(f"Runtime prompt skill '{style_name}' not found in prompt_skills/.")
        return skill_map[style_name].read_text(encoding="utf-8", errors="replace").strip()

    def list_llm_models(self) -> List[str]:
        llm_map, _ = self.discover_gguf_files()
        names = sorted(llm_map.keys())
        return names if names else ["No GGUF models found in models/LLM"]

    def list_mmproj_models(self) -> List[str]:
        _, mmproj_map = self.discover_gguf_files()
        return ["none"] + sorted(mmproj_map.keys())

    def list_system_prompts(self) -> List[str]:
        return ["none"] + sorted(self.discover_system_prompts().keys())

    def resolve_model_path(self, model_name: str, is_mmproj: bool = False) -> Optional[Path]:
        if not model_name or model_name.lower() in ("none", "no gguf models found in models/llm"):
            return None
        llm_map, mmproj_map = self.discover_gguf_files()
        lookup = mmproj_map if is_mmproj else llm_map
        if model_name in lookup:
            return lookup[model_name]
        for path in lookup.values():
            if path.name == model_name:
                return path
        raise FileNotFoundError(
            f"{'Vision projector (mmproj)' if is_mmproj else 'GGUF model'} '{model_name}' not found in LLM model directories."
        )

    def load_system_prompt_text(self, preset_name: str) -> Optional[str]:
        if not preset_name or preset_name.lower() == "none":
            return None
        prompt_map = self.discover_system_prompts()
        if preset_name not in prompt_map:
            raise FileNotFoundError(f"System prompt preset '{preset_name}' not found in models/LLM/prompts.")
        return prompt_map[preset_name].read_text(encoding="utf-8", errors="replace").strip()

    # ------------------------------------------------------------------
    # Binary Discovery, Manual Override & Version Inspection
    # ------------------------------------------------------------------

    def _load_kernel_settings(self) -> Dict[str, Any]:
        if not KERNEL_SETTINGS_FILE.is_file():
            return {}
        try:
            return json.loads(KERNEL_SETTINGS_FILE.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def _save_kernel_settings(self, data: Dict[str, Any]) -> None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        KERNEL_SETTINGS_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def get_custom_binary_path(self) -> Optional[str]:
        val = (self._load_kernel_settings().get("custom_binary_path") or "").strip()
        if not val:
            return None
        try:
            resolved = resolve_user_path(val)
            return str(resolved)
        except OSError:
            return val

    def get_custom_binary_display(self) -> Optional[str]:
        val = (self._load_kernel_settings().get("custom_binary_path") or "").strip()
        if not val:
            return None
        return sanitize_display_path(resolve_user_path(val))

    async def set_custom_binary_path(self, raw_path: str) -> Dict[str, Any]:
        """Set or clear a user-specified llama-server executable path (or folder containing it)."""
        cleaned = (raw_path or "").strip().strip("\"'")
        settings = self._load_kernel_settings()

        if not cleaned:
            settings.pop("custom_binary_path", None)
            self._save_kernel_settings(settings)
            self.unload()
            binary, source = self.find_binary_with_source()
            return {
                "ok": True,
                "reset": True,
                "binary_path": sanitize_display_path(binary),
                "binary_source": source,
                "local_version": await self.get_local_version(),
            }

        target = resolve_user_path(cleaned)
        exe_name = _executable_name()
        if target.is_dir():
            direct = target / exe_name
            if direct.is_file():
                target = direct
            else:
                matches = [p for p in target.rglob(exe_name) if p.is_file()]
                if not matches:
                    raise FileNotFoundError(f"Directory does not contain {exe_name}: {sanitize_display_path(target)}")
                target = matches[0]

        if not target.is_file():
            raise FileNotFoundError(f"Executable not found: {sanitize_display_path(target)}")
        if not target.name.lower().startswith("llama-server"):
            raise ValueError(f"Expected a llama-server executable ({exe_name}), got: {target.name}")

        resolved = target.resolve()
        self.unload()
        # Store sanitized relative token when inside ComfyUI root so cache files never leak local user paths
        settings["custom_binary_path"] = sanitize_display_path(resolved) or str(resolved)
        self._save_kernel_settings(settings)
        local_ver = await self.get_local_version()
        disp = sanitize_display_path(resolved)
        logger.info("[PromptAgent] Switched to custom llama-server binary: %s (%s)", disp, local_ver)
        return {
            "ok": True,
            "binary_path": disp,
            "binary_source": "custom_path",
            "local_version": local_ver or "unknown",
        }

    async def install_from_local_archives(self, archive_files: List[Tuple[str, bytes]]) -> Dict[str, Any]:
        """Extract one or more user-uploaded official llama.cpp .zip/.tar.gz archives into bin/."""
        if not archive_files:
            raise ValueError("No archive files provided.")

        self.unload()
        with tempfile.TemporaryDirectory(prefix="llama_cpp_manual_") as tmp_str:
            tmp_dir = Path(tmp_str)
            staging_dir = tmp_dir / "extracted"
            staging_dir.mkdir(parents=True, exist_ok=True)

            saved_names: List[str] = []
            for orig_name, content in archive_files:
                safe_name = Path(orig_name or "archive.zip").name
                low = safe_name.lower()
                if not (low.endswith(".zip") or low.endswith(".tar.gz") or low.endswith(".tgz")):
                    raise ValueError(f"Unsupported archive format '{safe_name}'. Please upload .zip or .tar.gz files.")
                dest_arc = tmp_dir / safe_name
                dest_arc.write_bytes(content)
                await asyncio.to_thread(_safe_extract_archive, dest_arc, staging_dir)
                saved_names.append(safe_name)

            exe_path = await asyncio.to_thread(_flatten_extracted_binaries, staging_dir, BIN_DIR)

        settings = self._load_kernel_settings()
        if settings.pop("custom_binary_path", None) is not None:
            self._save_kernel_settings(settings)

        local_ver = await self.get_local_version()
        VERSION_META_FILE.write_text(
            json.dumps(
                {
                    "tag": local_ver or "manual",
                    "binary_tag": local_ver or "manual",
                    "backend": "manual_archive",
                    "installed_at": time.time(),
                    "files": saved_names,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        disp = sanitize_display_path(exe_path)
        logger.info("[PromptAgent] Installed manual llama.cpp archive(s) %s -> %s (%s)", saved_names, disp, local_ver)
        return {
            "ok": True,
            "installed_version": local_ver or "unknown",
            "backend": "manual_archive",
            "binary_path": disp,
            "files": saved_names,
        }

    async def install_from_local_paths(self, raw_paths: List[str]) -> Dict[str, Any]:
        """Extract one or more local .zip/.tar.gz file paths (no HTTP upload size limit) into bin/."""
        archive_paths: List[Path] = []
        for raw in raw_paths:
            cleaned = (raw or "").strip().strip("\"'")
            if not cleaned:
                continue
            p = resolve_user_path(cleaned)
            if not p.exists():
                raise FileNotFoundError(f"Path not found: {sanitize_display_path(p)}")
            if p.is_dir():
                found = [
                    f for f in sorted(p.iterdir())
                    if f.is_file()
                    and (f.name.lower().endswith(".zip") or f.name.lower().endswith(".tar.gz") or f.name.lower().endswith(".tgz"))
                    and ("llama" in f.name.lower() or "cudart" in f.name.lower())
                ]
                if not found:
                    raise FileNotFoundError(f"No llama.cpp / cudart archive files found in folder: {sanitize_display_path(p)}")
                archive_paths.extend(found)
            elif p.is_file():
                low = p.name.lower()
                if not (low.endswith(".zip") or low.endswith(".tar.gz") or low.endswith(".tgz")):
                    raise ValueError(f"Unsupported archive format '{p.name}'. Provide .zip or .tar.gz files.")
                archive_paths.append(p)

        if not archive_paths:
            raise ValueError("Please enter at least one valid local .zip / .tar.gz path.")

        self.unload()
        with tempfile.TemporaryDirectory(prefix="llama_cpp_local_") as tmp_str:
            staging_dir = Path(tmp_str) / "extracted"
            staging_dir.mkdir(parents=True, exist_ok=True)
            for arc in archive_paths:
                await asyncio.to_thread(_safe_extract_archive, arc, staging_dir)
            exe_path = await asyncio.to_thread(_flatten_extracted_binaries, staging_dir, BIN_DIR)

        settings = self._load_kernel_settings()
        if settings.pop("custom_binary_path", None) is not None:
            self._save_kernel_settings(settings)

        local_ver = await self.get_local_version()
        names = [p.name for p in archive_paths]
        VERSION_META_FILE.write_text(
            json.dumps(
                {
                    "tag": local_ver or "manual",
                    "binary_tag": local_ver or "manual",
                    "backend": "manual_archive",
                    "installed_at": time.time(),
                    "files": names,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        disp = sanitize_display_path(exe_path)
        logger.info("[PromptAgent] Installed local llama.cpp archive(s) %s -> %s (%s)", names, disp, local_ver)
        return {
            "ok": True,
            "installed_version": local_ver or "unknown",
            "backend": "manual_archive",
            "binary_path": disp,
            "files": names,
        }

    def find_binary_with_source(self) -> Tuple[Optional[Path], Optional[str]]:
        exe_name = _executable_name()

        # 1. Explicit user-configured custom binary path
        custom_path = self.get_custom_binary_path()
        if custom_path and Path(custom_path).is_file():
            return Path(custom_path).resolve(), "custom_path"

        # 2. Managed bin/ inside this custom node
        local_bin = BIN_DIR / exe_name
        if local_bin.is_file():
            return local_bin.resolve(), "local_bin"
        if BIN_DIR.is_dir():
            for match in BIN_DIR.rglob(exe_name):
                if match.is_file():
                    return match.resolve(), "local_bin"

        # 3. Explicit env override
        env_bin = os.environ.get("LLAMA_SERVER_BIN")
        if env_bin and Path(env_bin).is_file():
            return Path(env_bin).resolve(), "env"

        # 4. System PATH
        system_bin = shutil.which("llama-server")
        if system_bin:
            return Path(system_bin).resolve(), "system_path"

        return None, None

    def find_binary(self) -> Optional[Path]:
        path, _ = self.find_binary_with_source()
        return path

    def _read_version_meta(self) -> Optional[Dict[str, Any]]:
        if not VERSION_META_FILE.is_file():
            return None
        try:
            return json.loads(VERSION_META_FILE.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    async def get_local_version(self) -> Optional[str]:
        if self.is_running() and self._port:
            try:
                timeout = aiohttp.ClientTimeout(total=2)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.get(f"http://127.0.0.1:{self._port}/props") as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            parsed = _parse_version_from_cli_output(str(data.get("build_info") or ""))
                            if parsed:
                                return parsed
            except Exception:
                pass

        binary, source = self.find_binary_with_source()
        if binary is None:
            return None

        def _run_version_cmd() -> Optional[str]:
            kwargs: Dict[str, Any] = {
                "stdout": subprocess.PIPE,
                "stderr": subprocess.PIPE,
                "text": True,
                "timeout": 5,
                "cwd": str(binary.parent),
            }
            if _is_windows():
                kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
            try:
                proc = subprocess.run([str(binary), "--version"], **kwargs)
                combined = f"{proc.stdout or ''}\n{proc.stderr or ''}"
                return _parse_version_from_cli_output(combined)
            except Exception:
                return None

        cli_ver = await asyncio.to_thread(_run_version_cmd)

        if source == "local_bin":
            meta = self._read_version_meta()
            if meta and meta.get("tag"):
                if cli_ver is None:
                    return _normalize_version_tag(meta.get("binary_tag") or meta["tag"])
                if meta.get("binary_tag") and _normalize_version_tag(meta["binary_tag"]) == _normalize_version_tag(cli_ver):
                    return _normalize_version_tag(meta["tag"])

        return cli_ver

    # ------------------------------------------------------------------
    # Process Lifecycle & VRAM Management
    # ------------------------------------------------------------------

    def is_running(self) -> bool:
        with self._proc_lock:
            return self._process is not None and self._process.poll() is None

    def get_status(self) -> Dict[str, Any]:
        binary, source = self.find_binary_with_source()
        with self._proc_lock:
            running = self._process is not None and self._process.poll() is None
            cfg = self._current_config if running else None
            return {
                "running": running,
                "port": self._port if running else None,
                "model": Path(cfg[0]).name if cfg else None,
                "mmproj": Path(cfg[1]).name if (cfg and cfg[1]) else None,
                "binary_installed": binary is not None,
                "binary_path": sanitize_display_path(binary),
                "binary_source": source,
                "auto_unload_on_queue": self.auto_unload_on_queue,
            }

    def unload(self) -> bool:
        """Terminate llama-server immediately so the OS releases 100% of its VRAM."""
        with self._proc_lock:
            proc = self._process
            log_fh = self._log_fh
            self._process = None
            self._log_fh = None
            self._current_config = None
            self._port = None

            if proc is not None and proc.poll() is None:
                logger.info("[PromptAgent] Stopping llama-server (PID %s) to release VRAM...", proc.pid)
                try:
                    proc.terminate()
                    proc.wait(timeout=4)
                except subprocess.TimeoutExpired:
                    logger.warning("[PromptAgent] llama-server did not exit cleanly; forcing kill.")
                    try:
                        proc.kill()
                        proc.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        if _is_windows():
                            subprocess.run(
                                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                                creationflags=subprocess.CREATE_NO_WINDOW,
                            )
                except OSError:
                    pass

            # Close log file handle only AFTER child process has exited
            if log_fh is not None:
                try:
                    log_fh.close()
                except OSError:
                    pass

            return proc is not None

    def _read_recent_server_log(self, max_chars: int = 1600) -> str:
        if not SERVER_LOG_FILE.is_file():
            return ""
        try:
            text = SERVER_LOG_FILE.read_text(encoding="utf-8", errors="replace")
            return text[-max_chars:].strip()
        except OSError:
            return ""

    async def ensure_running(
        self,
        model_path: Path,
        mmproj_path: Optional[Path] = None,
        ctx_size: int = 8192,
        memory_mode: str = "auto",
        n_gpu_layers: int = 99,
        n_cpu_moe_layers: int = 1,
        reasoning: str = "off",
        extra_args: str = "",
        startup_timeout: float = 120.0,
    ) -> int:
        """Ensure llama-server is running with the requested model, mmproj, and memory settings."""
        await asyncio.to_thread(self._lifecycle_lock.acquire)
        try:
            binary = self.find_binary()
            if binary is None:
                raise RuntimeError(
                    "llama-server executable not found. Click '⚙ Kernel' in the Prompt Agent toolbar to download llama.cpp into bin/, or set LLAMA_SERVER_BIN."
                )

            parsed_extra = tuple(split_extra_args(extra_args))
            clean_memory_mode = memory_mode if memory_mode in {
                "auto", "gpu_layers", "cpu_moe_layers", "gpu_and_cpu_moe_layers"
            } else "auto"
            clean_reasoning = reasoning if reasoning in {"auto", "on", "off"} else "off"

            desired = (
                str(model_path.resolve()),
                str(mmproj_path.resolve()) if mmproj_path else None,
                int(ctx_size),
                clean_memory_mode,
                int(n_gpu_layers),
                int(n_cpu_moe_layers),
                clean_reasoning,
                parsed_extra,
            )

            if self.is_running() and self._current_config == desired and self._port is not None:
                if await self._check_health(self._port):
                    return self._port

            self.unload()

            port = _pick_free_port(8088)
            cmd = [
                str(binary),
                "-m", desired[0],
                "--host", "127.0.0.1",
                "--port", str(port),
                "--jinja",
                "-c", str(desired[2]),
                "--reasoning", clean_reasoning,
            ]

            if clean_memory_mode in {"gpu_layers", "gpu_and_cpu_moe_layers"}:
                cmd.extend(["-ngl", str(desired[4])])
            if clean_memory_mode in {"cpu_moe_layers", "gpu_and_cpu_moe_layers"}:
                cmd.extend(["--n-cpu-moe", str(desired[5])])

            if desired[1]:
                cmd.extend(["--mmproj", desired[1]])

            if parsed_extra:
                cmd.extend(parsed_extra)

            logger.info("[PromptAgent] Launching llama-server on port %d: %s", port, Path(desired[0]).name)

            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            log_fh = open(SERVER_LOG_FILE, "w", encoding="utf-8", errors="replace")
            popen_kwargs: Dict[str, Any] = {
                "stdout": log_fh,
                "stderr": subprocess.STDOUT,
                "cwd": str(binary.parent),
            }
            if _is_windows():
                popen_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

            with self._proc_lock:
                self._log_fh = log_fh
                self._process = subprocess.Popen(cmd, **popen_kwargs)
                self._port = port
                self._current_config = desired

            deadline = time.monotonic() + startup_timeout
            while time.monotonic() < deadline:
                if not self.is_running():
                    err_tail = self._read_recent_server_log()
                    friendly = _parse_llama_error(err_tail)
                    self.unload()
                    if friendly:
                        raise RuntimeError(friendly)
                    raise RuntimeError(f"llama-server exited unexpectedly during startup.\n{err_tail}")
                if await self._check_health(port):
                    logger.info("[PromptAgent] llama-server ready on 127.0.0.1:%d", port)
                    return port
                await asyncio.sleep(0.5)

            err_tail = self._read_recent_server_log()
            friendly = _parse_llama_error(err_tail)
            self.unload()
            raise TimeoutError(friendly or f"Timed out waiting for llama-server to load model.\n{err_tail}")
        finally:
            self._lifecycle_lock.release()

    async def _check_health(self, port: int) -> bool:
        try:
            timeout = aiohttp.ClientTimeout(total=2)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(f"http://127.0.0.1:{port}/health") as resp:
                    if resp.status == 200:
                        try:
                            data = await resp.json()
                            status = str(data.get("status", "ok")).lower()
                            return status in ("ok", "ready")
                        except Exception:
                            return True
                    return False
        except Exception:
            return False

    # ------------------------------------------------------------------
    # OpenAI-Compatible HTTP Client (with reasoning & perf extraction)
    # ------------------------------------------------------------------

    async def chat_completions(
        self,
        messages: List[Dict[str, Any]],
        response_format: Optional[Dict[str, Any]] = None,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
        repeat_penalty: float = 1.0,
        max_tokens: int = 2048,
        seed: int = -1,
        timeout_seconds: int = 300,
    ) -> Dict[str, str]:
        """Return {'content': ..., 'reasoning': ..., 'perf': ...} from llama-server."""
        if not self.is_running() or not self._port:
            raise RuntimeError("llama-server is not running.")

        payload: Dict[str, Any] = {
            "model": "local-model",
            "messages": messages,
            "temperature": float(temperature),
            "top_p": float(top_p),
            "top_k": int(top_k),
            "repeat_penalty": float(repeat_penalty),
            "max_tokens": int(max_tokens),
            "seed": normalize_llama_seed(seed),
            "stream": False,
        }
        if response_format is not None:
            payload["response_format"] = response_format

        url = f"http://127.0.0.1:{self._port}/v1/chat/completions"
        timeout = aiohttp.ClientTimeout(total=max(30, int(timeout_seconds)))
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload) as resp:
                body_text = await resp.text()
                if resp.status != 200:
                    if response_format is not None and resp.status in (400, 422, 500):
                        logger.warning(
                            "[PromptAgent] llama-server returned %d with response_format; retrying without it.",
                            resp.status,
                        )
                        payload.pop("response_format", None)
                        async with session.post(url, json=payload) as retry_resp:
                            retry_text = await retry_resp.text()
                            if retry_resp.status != 200:
                                raise RuntimeError(f"llama-server error ({retry_resp.status}): {retry_text[:500]}")
                            data = json.loads(retry_text)
                    else:
                        raise RuntimeError(f"llama-server error ({resp.status}): {body_text[:500]}")
                else:
                    data = json.loads(body_text)

        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"Empty choices returned by llama-server: {data}")

        message = choices[0].get("message") or {}
        content = str(message.get("content") or "").strip()
        reasoning = str(message.get("reasoning_content") or "").strip()

        # Extract inline <think>...</think> if reasoning_content wasn't separated by server
        if not reasoning and "<think>" in content.lower():
            think_matches = re.findall(r"<think>(.*?)</think>", content, flags=re.DOTALL | re.IGNORECASE)
            if think_matches:
                reasoning = "\n\n".join(m.strip() for m in think_matches if m.strip())
                content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL | re.IGNORECASE).strip()

        perf = _format_timings(data.get("timings"))
        return {
            "content": content,
            "reasoning": reasoning,
            "perf": perf,
        }

    # ------------------------------------------------------------------
    # GitHub Release Version Check (v1.0.0 nightly-tag.txt + Web Fallback)
    # ------------------------------------------------------------------

    def _load_release_cache(self) -> Dict[str, Any]:
        if not RELEASE_CACHE_FILE.is_file():
            return {}
        try:
            return json.loads(RELEASE_CACHE_FILE.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def _save_release_cache(self, data: Dict[str, Any]) -> None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        RELEASE_CACHE_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def skip_version(self, version_tag: str) -> Dict[str, Any]:
        cache = self._load_release_cache()
        norm = _normalize_version_tag(version_tag) or version_tag
        cache["skipped_version"] = norm
        self._save_release_cache(cache)
        return {"ok": True, "skipped_version": norm}

    async def _fetch_expanded_assets(self, session: aiohttp.ClientSession, tag: str) -> List[Dict[str, Any]]:
        """Scrape GitHub's non-API expanded_assets HTML fragment (zero API rate limit)."""
        url = f"{GITHUB_EXPANDED_ASSETS_BASE}/{tag}"
        headers = {"User-Agent": "ComfyUI-Prompt-Agent"}
        async with session.get(url, headers=headers) as resp:
            if resp.status != 200:
                return []
            html = await resp.text(errors="replace")

        assets: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for rel_href in _EXPANDED_HREF_RE.findall(html):
            filename = unquote(rel_href.rsplit("/", 1)[-1])
            if not filename or filename in seen:
                continue
            seen.add(filename)
            assets.append(
                {
                    "name": filename,
                    "browser_download_url": f"https://github.com{rel_href}",
                    "size": 0,
                }
            )
        return assets

    async def _resolve_release_binaries(
        self,
        session: aiohttp.ClientSession,
        release_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Follow v1.0.0 nightly-tag.txt indirection and ensure platform assets are populated."""
        tag = release_data.get("tag_name") or ""
        assets = release_data.get("assets") or []

        if _has_platform_binary_archives(assets):
            release_data["binary_tag"] = release_data.get("binary_tag") or tag
            return release_data

        # Check for nightly-tag.txt indirection (used by ggml-org/llama.cpp v1.0.0+)
        binary_tag = release_data.get("binary_tag") or ""
        nightly_url = None
        for a in assets:
            if (a.get("name") or "").lower() == "nightly-tag.txt":
                nightly_url = a.get("browser_download_url")
                break
        if not nightly_url and tag:
            nightly_url = f"{GITHUB_DOWNLOAD_BASE}/{tag}/nightly-tag.txt"

        if not binary_tag and nightly_url:
            try:
                async with session.get(nightly_url, headers={"User-Agent": "ComfyUI-Prompt-Agent"}) as resp:
                    if resp.status == 200:
                        txt = (await resp.text()).strip()
                        if txt and len(txt) <= 32 and "\n" not in txt:
                            binary_tag = txt
                            logger.info("[PromptAgent] Resolved %s nightly-tag.txt -> %s", tag, binary_tag)
            except Exception as exc:
                logger.warning("[PromptAgent] Could not fetch nightly-tag.txt: %s", exc)

        target_tag = binary_tag or tag
        if target_tag:
            expanded = await self._fetch_expanded_assets(session, target_tag)
            if expanded:
                release_data["assets"] = expanded
                release_data["binary_tag"] = target_tag

        return release_data

    async def _fetch_latest_release_metadata(self) -> Dict[str, Any]:
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "ComfyUI-Prompt-Agent",
        }
        timeout = aiohttp.ClientTimeout(total=20)
        release_data: Optional[Dict[str, Any]] = None
        api_error: Optional[str] = None

        async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as session:
            # 1. Primary: GitHub REST API (gives full release notes + byte sizes)
            try:
                async with session.get(GITHUB_LATEST_RELEASE_API, headers=headers) as resp:
                    if resp.status == 200:
                        raw = await resp.json()
                        release_data = {
                            "tag_name": raw.get("tag_name", ""),
                            "binary_tag": "",
                            "name": raw.get("name") or raw.get("tag_name", ""),
                            "body": raw.get("body") or "",
                            "html_url": raw.get("html_url") or GITHUB_LATEST_RELEASE_WEB,
                            "published_at": raw.get("published_at") or "",
                            "assets": [
                                {
                                    "name": a.get("name", ""),
                                    "browser_download_url": a.get("browser_download_url", ""),
                                    "size": a.get("size", 0),
                                }
                                for a in (raw.get("assets") or [])
                            ],
                        }
                    else:
                        api_error = f"GitHub API HTTP {resp.status}"
            except Exception as exc:
                api_error = str(exc)

            # 2. Fallback when GitHub API is rate-limited (HTTP 403): resolve via web redirect
            if release_data is None:
                logger.info("[PromptAgent] Falling back to GitHub web release endpoint (%s).", api_error)
                async with session.get(
                    GITHUB_LATEST_RELEASE_WEB,
                    headers={"User-Agent": "ComfyUI-Prompt-Agent"},
                    allow_redirects=True,
                ) as resp:
                    if resp.status != 200:
                        raise RuntimeError(f"Failed to query GitHub releases ({api_error}; web HTTP {resp.status})")
                    final_url = str(resp.url)
                    tag_match = re.search(r"/releases/tag/([^/?#]+)", final_url)
                    if not tag_match:
                        raise RuntimeError(f"Could not parse latest release tag from {final_url}")
                    tag_name = unquote(tag_match.group(1))
                    release_data = {
                        "tag_name": tag_name,
                        "binary_tag": "",
                        "name": tag_name,
                        "body": f"Latest release {tag_name} resolved via github.com (GitHub API rate-limited: {api_error}).",
                        "html_url": final_url,
                        "published_at": "",
                        "assets": [],
                    }

            # 3. Resolve nightly-tag.txt indirection & binary assets
            release_data = await self._resolve_release_binaries(session, release_data)
            return release_data

    async def check_for_updates(self, force: bool = False) -> Dict[str, Any]:
        cache = self._load_release_cache()
        now = time.time()
        used_cache = False
        release_data = cache.get("release")

        # Auto-heal stale caches that only stored nightly-tag.txt without actual platform binaries
        cache_has_binaries = bool(release_data and _has_platform_binary_archives(release_data.get("assets") or []))

        if (
            not force
            and release_data
            and cache_has_binaries
            and (now - float(cache.get("checked_at", 0))) < RELEASE_CACHE_TTL_SECONDS
        ):
            used_cache = True
        else:
            try:
                release_data = await self._fetch_latest_release_metadata()
                cache["checked_at"] = now
                cache["release"] = release_data
                self._save_release_cache(cache)
            except Exception as exc:
                if release_data and cache_has_binaries:
                    used_cache = True
                    logger.warning("[PromptAgent] Release check failed (%s); using cached metadata.", exc)
                else:
                    binary, source = self.find_binary_with_source()
                    return {
                        "ok": False,
                        "error": str(exc),
                        "installed": binary is not None,
                        "binary_path": sanitize_display_path(binary),
                        "binary_source": source,
                        "custom_binary_path": self.get_custom_binary_display(),
                        "bin_dir": sanitize_display_path(BIN_DIR),
                        "hardware": detect_gpu_hardware(),
                        "official_releases_url": GITHUB_RELEASES_WEB,
                        "binary_release_url": GITHUB_RELEASES_WEB,
                        "local_version": await self.get_local_version(),
                        "update_available": False,
                    }

        binary, source = self.find_binary_with_source()
        local_version = await self.get_local_version()
        latest_tag = release_data.get("tag_name", "")
        binary_tag = release_data.get("binary_tag") or latest_tag
        latest_norm = _normalize_version_tag(latest_tag) or latest_tag
        binary_norm = _normalize_version_tag(binary_tag) or binary_tag

        skipped_version = cache.get("skipped_version")
        update_available = (binary is None) or is_newer_version(local_version, latest_norm, binary_norm)
        is_skipped = bool(
            skipped_version
            and _normalize_version_tag(skipped_version) in {latest_norm, binary_norm}
        )

        asset_match = match_release_assets(release_data.get("assets") or [])
        matched_summary = {
            k: {
                "label": v["label"],
                "files": [v["primary"]["name"]] + [e["name"] for e in v["extras"]],
                "downloads": [
                    {"name": v["primary"]["name"], "url": v["primary"].get("browser_download_url", "")}
                ]
                + [
                    {"name": e["name"], "url": e.get("browser_download_url", "")}
                    for e in v["extras"]
                ],
                "total_size": int(v["primary"].get("size", 0)) + sum(int(e.get("size", 0)) for e in v["extras"]),
            }
            for k, v in asset_match["matched"].items()
        }

        notes = release_data.get("body") or ""
        if len(notes) > 2500:
            notes = notes[:2500].rstrip() + "\n..."

        display_latest = (
            f"{latest_norm} ({binary_norm})"
            if binary_norm and binary_norm != latest_norm
            else latest_norm
        )
        binary_release_url = f"{GITHUB_TAG_WEB_BASE}/{binary_tag}" if binary_tag else GITHUB_RELEASES_WEB

        return {
            "ok": True,
            "installed": binary is not None,
            "binary_path": sanitize_display_path(binary),
            "binary_source": source,
            "custom_binary_path": self.get_custom_binary_display(),
            "bin_dir": sanitize_display_path(BIN_DIR),
            "local_version": local_version or ("not_installed" if binary is None else "unknown"),
            "latest_version": latest_norm,
            "binary_tag": binary_norm,
            "latest_display": display_latest,
            "latest_tag_raw": latest_tag,
            "release_name": release_data.get("name", latest_tag),
            "release_notes": notes,
            "published_at": release_data.get("published_at", ""),
            "html_url": release_data.get("html_url", "") or GITHUB_LATEST_RELEASE_WEB,
            "binary_release_url": binary_release_url,
            "official_releases_url": GITHUB_RELEASES_WEB,
            "update_available": update_available,
            "skipped": is_skipped,
            "skipped_version": skipped_version,
            "cached": used_cache,
            "hardware": asset_match["hardware"],
            "recommended_backend": asset_match["recommended_backend"],
            "recommendation_reason": asset_match["recommendation_reason"],
            "available_backends": asset_match["available_backends"],
            "matched_assets": matched_summary,
        }

    def get_update_progress(self) -> Dict[str, Any]:
        return dict(self._update_job)

    def _set_update_job(self, **updates: Any) -> None:
        self._update_job.update(updates)

    def start_update_task(self, backend: str = "auto") -> Dict[str, Any]:
        """Start kernel download/install as a background asyncio task so HTTP never times out."""
        if self._update_task is not None and not self._update_task.done():
            return {"ok": True, "already_running": True, "job": self.get_update_progress()}

        self._update_job = {
            "state": "starting",
            "stage": "Preparing llama.cpp kernel download...",
            "backend": backend,
            "current_file": "",
            "file_index": 0,
            "file_count": 0,
            "downloaded_bytes": 0,
            "total_bytes": 0,
            "percent": 0.0,
            "speed_bps": 0.0,
            "error": None,
            "result": None,
        }

        async def _runner() -> None:
            try:
                res = await self.download_and_install_update(backend=backend)
                self._set_update_job(
                    state="done",
                    stage=f"Installed llama.cpp {res.get('installed_version', '')} ({res.get('backend', '')}) successfully!",
                    percent=100.0,
                    error=None,
                    result=res,
                )
            except asyncio.CancelledError:
                self._set_update_job(
                    state="cancelled",
                    stage="Download cancelled by user.",
                    error=None,
                    result=None,
                )
            except Exception as exc:
                logger.exception("[PromptAgent] Background kernel update failed")
                self._set_update_job(
                    state="error",
                    stage=f"Update failed: {exc}",
                    error=str(exc),
                    result=None,
                )

        self._update_task = asyncio.create_task(_runner())
        return {"ok": True, "started": True, "job": self.get_update_progress()}

    def cancel_update_task(self) -> Dict[str, Any]:
        if self._update_task is not None and not self._update_task.done():
            self._update_task.cancel()
            return {"ok": True, "cancelled": True}
        return {"ok": True, "cancelled": False}

    async def download_and_install_update(self, backend: str = "auto") -> Dict[str, Any]:
        """Download and extract the matched llama.cpp release archive into bin/ with live progress."""
        self._set_update_job(state="starting", stage="Resolving release metadata...", error=None)
        cache = self._load_release_cache()
        release_data = cache.get("release")
        if not release_data or not _has_platform_binary_archives(release_data.get("assets") or []):
            check_res = await self.check_for_updates(force=True)
            if not check_res.get("ok"):
                raise RuntimeError(check_res.get("error", "Unable to fetch latest llama.cpp release metadata."))
            cache = self._load_release_cache()
            release_data = cache.get("release")

        asset_match = match_release_assets(release_data.get("assets") or [])
        matched = asset_match["matched"]
        if not matched:
            raise RuntimeError(
                f"No compatible prebuilt llama.cpp archive found for platform {asset_match['platform']['os']}/{asset_match['platform']['arch']}."
            )

        selected_backend = asset_match["recommended_backend"] if backend in ("auto", "", None) else backend
        if selected_backend not in matched:
            raise ValueError(
                f"Backend '{selected_backend}' is not available in this release. Available: {list(matched.keys())}"
            )

        bundle = matched[selected_backend]
        downloads = [bundle["primary"]] + list(bundle["extras"])
        file_count = len(downloads)

        # Stop running server before replacing binaries (avoids Windows file-in-use lock)
        self.unload()

        proxy_url = _get_proxy_url()
        with tempfile.TemporaryDirectory(prefix="llama_cpp_update_") as tmp_str:
            tmp_dir = Path(tmp_str)
            staging_dir = tmp_dir / "extracted"
            staging_dir.mkdir(parents=True, exist_ok=True)

            # No total cap so large archives over slow links never hit an artificial total timeout;
            # sock_read=120 detects truly stalled sockets cleanly.
            timeout = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=120)
            headers = {"User-Agent": "ComfyUI-Prompt-Agent"}

            async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as session:
                for idx, asset in enumerate(downloads, start=1):
                    url = asset["browser_download_url"]
                    filename = asset["name"]
                    archive_path = tmp_dir / filename

                    self._set_update_job(
                        state="downloading",
                        backend=selected_backend,
                        current_file=filename,
                        file_index=idx,
                        file_count=file_count,
                        downloaded_bytes=0,
                        total_bytes=int(asset.get("size") or 0),
                        percent=0.0,
                        speed_bps=0.0,
                        stage=f"Downloading {filename} ({idx}/{file_count})...",
                    )
                    logger.info("[PromptAgent] Downloading %s (%d/%d)...", filename, idx, file_count)

                    last_err: Optional[Exception] = None
                    for attempt in range(1, 4):
                        try:
                            req_kwargs: Dict[str, Any] = {"headers": headers, "allow_redirects": True}
                            if proxy_url:
                                req_kwargs["proxy"] = proxy_url
                            async with session.get(url, **req_kwargs) as resp:
                                if resp.status != 200:
                                    raise RuntimeError(f"Failed to download {filename}: HTTP {resp.status}")
                                content_len = int(resp.headers.get("Content-Length") or asset.get("size") or 0)
                                downloaded = 0
                                started_at = time.monotonic()
                                last_log = started_at
                                self._set_update_job(total_bytes=content_len)

                                with open(archive_path, "wb") as fh:
                                    async for chunk in resp.content.iter_chunked(1024 * 256):
                                        fh.write(chunk)
                                        downloaded += len(chunk)
                                        now = time.monotonic()
                                        elapsed = max(now - started_at, 0.001)
                                        speed = downloaded / elapsed
                                        pct = (downloaded / content_len * 100.0) if content_len > 0 else 0.0
                                        self._set_update_job(
                                            downloaded_bytes=downloaded,
                                            total_bytes=content_len,
                                            percent=round(pct, 1),
                                            speed_bps=speed,
                                        )
                                        if now - last_log >= 2.0:
                                            last_log = now
                                            logger.info(
                                                "[PromptAgent] %s: %.1f%% (%.1f MB/s)",
                                                filename,
                                                pct,
                                                speed / (1024 * 1024),
                                            )
                            last_err = None
                            break
                        except asyncio.CancelledError:
                            raise
                        except Exception as exc:
                            last_err = exc
                            if attempt < 3:
                                logger.warning("[PromptAgent] Download retry %d/3 for %s: %s", attempt, filename, exc)
                                self._set_update_job(
                                    stage=f"Retrying {filename} (attempt {attempt + 1}/3)..."
                                )
                                await asyncio.sleep(2.0)
                    if last_err is not None:
                        raise last_err

                    self._set_update_job(
                        state="extracting",
                        stage=f"Extracting {filename} ({idx}/{file_count})...",
                        percent=100.0,
                    )
                    await asyncio.to_thread(_safe_extract_archive, archive_path, staging_dir)

            self._set_update_job(state="extracting", stage="Installing binaries into bin/...")
            exe_path = await asyncio.to_thread(_flatten_extracted_binaries, staging_dir, BIN_DIR)

        tag_raw = release_data.get("tag_name", "")
        tag_norm = _normalize_version_tag(tag_raw) or tag_raw
        binary_tag = _normalize_version_tag(release_data.get("binary_tag") or tag_raw) or tag_norm

        VERSION_META_FILE.write_text(
            json.dumps(
                {
                    "tag": tag_norm,
                    "binary_tag": binary_tag,
                    "tag_raw": tag_raw,
                    "backend": selected_backend,
                    "installed_at": time.time(),
                    "files": [a["name"] for a in downloads],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        if cache.get("skipped_version") in {tag_norm, binary_tag}:
            cache.pop("skipped_version", None)
            self._save_release_cache(cache)

        settings = self._load_kernel_settings()
        if settings.pop("custom_binary_path", None) is not None:
            self._save_kernel_settings(settings)

        local_ver = await self.get_local_version()
        disp = sanitize_display_path(exe_path)
        logger.info("[PromptAgent] Installed llama.cpp %s (%s) to %s", local_ver or tag_norm, selected_backend, disp)
        return {
            "ok": True,
            "installed_version": local_ver or tag_norm,
            "binary_tag": binary_tag,
            "backend": selected_backend,
            "binary_path": disp,
        }


kernel_manager = LlamaServerManager()
