"""ComfyUI-Prompt-Agent — Standalone static-rule / dynamic-dataflow prompt pipeline node
and Universal Text & Reasoning Preview node.

Powered by a decoupled llama-server subprocess and built-in self-contained WD14 ONNX engine.
Zero dependency on external third-party custom nodes.
"""

from __future__ import annotations

import asyncio
import base64
import concurrent.futures
import gc
import io
import json
import logging
import math
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from aiohttp import web
from PIL import Image

try:
    import comfy.model_management
    import comfy.utils
except ImportError:
    comfy = None

try:
    from server import PromptServer
except ImportError:
    PromptServer = None

from .kernel_manager import MAX_LLAMA_SEED, kernel_manager
from .wd14_engine import (
    DEFAULT_WD14_EXCLUDE_TAGS,
    DEFAULT_WD14_MODEL,
    format_tiered_wd14_item,
    list_wd14_models,
    prepare_rgb_image,
    run_wd14_batch_groups,
)

logger = logging.getLogger("ComfyUI-Prompt-Agent")

WEB_DIRECTORY = "./web"
MAX_IMAGES = 6

# 7 Canonical Visual Prompt Slots
PROMPT_SLOT_KEYS: Tuple[str, ...] = (
    "subject",
    "clothing",
    "pose_expression",
    "background",
    "lighting",
    "composition",
    "style",
)

# Built-in Slot Preservation Presets (`preserve_slots`)
PRESERVE_SLOT_PRESETS: Dict[str, Tuple[str, ...]] = {
    "none": (),
    "outfit_pose_bg_lighting": ("clothing", "pose_expression", "background", "lighting", "composition"),
    "subject_pose_bg_lighting": ("subject", "pose_expression", "background", "lighting", "composition"),
    "subject_outfit_bg_lighting": ("subject", "clothing", "background", "lighting", "composition"),
    "subject_outfit_pose_bg": ("subject", "clothing", "pose_expression", "background", "composition"),
    "subject_only": ("subject",),
    "bg_and_lighting": ("background", "lighting", "composition"),
}

# Built-in Runtime Prompt Skills (`output_style`)
OUTPUT_STYLE_SKILLS: Dict[str, str] = {
    "auto": (
        "RUNTIME PROMPT SKILL [Auto Balanced]:\n"
        "- Produce a clean, concrete, ready-to-use English image generation prompt grounded in the 7 visual slots."
    ),
    "natural_language": (
        "RUNTIME PROMPT SKILL [Natural Language — Flux / Qwen-Image / HiDream]:\n"
        "- De-bag-of-words all WD14 tags into fluent, cohesive English photographic/artistic prose.\n"
        "- Follow natural visual flow: subject & character identity -> pose & gaze -> clothing fabrics & details -> "
        "composition & camera lens -> lighting direction & color palette -> environment background -> artistic medium.\n"
        "- NEVER output raw Danbooru tag lists like '1girl, solo, looking at viewer' in final_prompt.\n"
        "- Expand only with concrete visible physical details; ban empty filler buzzwords like 'masterpiece, breathtaking, hyperdetailed'."
    ),
    "danbooru_tags": (
        "RUNTIME PROMPT SKILL [Danbooru Tags — Illustrious / NoobAI / Pony / SDXL]:\n"
        "- Output clean comma-separated canonical Danbooru tags in final_prompt.\n"
        "- Order cleanly: subject/character -> hair/eyes/appearance -> clothing/accessories -> pose/expression -> "
        "background/setting -> lighting/atmosphere -> composition/camera -> art style.\n"
        "- Deduplicate redundant tags and strictly delete contradicting tags whenever an attribute changes."
    ),
    "hybrid": (
        "RUNTIME PROMPT SKILL [Hybrid — Anchored Tags + Natural Visual Prose]:\n"
        "- Start with concise high-confidence anchor tags for subject/character identity and art style, "
        "followed by clear natural-language sentences describing pose, clothing textures, spatial layout, lighting, and background."
    ),
    "edit_instruction": (
        "RUNTIME PROMPT SKILL [Image Edit Instruction — Flux Kontext / Qwen-Image-Edit]:\n"
        "- Write `final_prompt` as a direct, unambiguous, single-paragraph English image-editing instruction "
        "describing exactly what to modify and explicitly stating what unchanged elements (outfit, pose, lighting, background, style) must be preserved.\n"
        "- Example style: 'Transform the female character into an otokonoko with boyish facial features and a flat chest, "
        "while keeping the exact school uniform, hairstyle silhouette, standing pose, classroom background, and warm sunlight unchanged.'"
    ),
    "structured_slots": (
        "RUNTIME PROMPT SKILL [Structured Slots High-Fidelity]:\n"
        "- Prioritize maximally accurate, non-overlapping decomposition into the 7 visual slots, then assemble final_prompt cleanly from those slots."
    ),
    "custom_preset": (
        "RUNTIME PROMPT SKILL [Custom System Preset Priority]:\n"
        "- Strictly prioritize formatting, vocabulary, and stylistic directives from USER SYSTEM PROMPT PRESET "
        "while returning valid JSON containing summary, slots, and final_prompt."
    ),
}

# Universal Thinking / Reasoning Tag Patterns
_THINKING_PATTERNS: Tuple[re.Pattern[str], ...] = (
    re.compile(r"<think>(.*?)</think>", flags=re.DOTALL | re.IGNORECASE),
    re.compile(r"<thinking>(.*?)</thinking>", flags=re.DOTALL | re.IGNORECASE),
    re.compile(r"<reasoning>(.*?)</reasoning>", flags=re.DOTALL | re.IGNORECASE),
    re.compile(r"◁think▷(.*?)◁/think▷", flags=re.DOTALL | re.IGNORECASE),
    re.compile(r"\[Start thinking\](.*?)\[End thinking\]", flags=re.DOTALL | re.IGNORECASE),
)
_UNCLOSED_THINK_RE = re.compile(
    r"^(.*?)(?:<think>|<thinking>|<reasoning>|◁think▷|\[Start thinking\])(.*)$",
    flags=re.DOTALL | re.IGNORECASE,
)


def extract_thinking_from_text(raw_text: str) -> Tuple[str, str]:
    """Universal extractor returning (clean_text_without_thinking, extracted_thinking)."""
    text = str(raw_text or "")
    if not text.strip():
        return "", ""

    collected: List[str] = []
    cleaned = text
    for pat in _THINKING_PATTERNS:
        matches = pat.findall(cleaned)
        if matches:
            for m in matches:
                if m.strip():
                    collected.append(m.strip())
            cleaned = pat.sub("", cleaned)

    # Also handle truncated unclosed thinking tag (when generation hit max_tokens inside <think>)
    if not collected:
        m_unclosed = _UNCLOSED_THINK_RE.match(cleaned)
        if m_unclosed:
            prefix = m_unclosed.group(1).strip()
            thought = m_unclosed.group(2).strip()
            if thought:
                collected.append(thought)
            cleaned = prefix

    return cleaned.strip(), "\n\n".join(collected).strip()


# ======================================================================
# Helpers: 7-Slot Normalization, Image Conversion & JSON Parsing
# ======================================================================

def _empty_slots() -> Dict[str, str]:
    return {k: "" for k in PROMPT_SLOT_KEYS}


def _normalize_slots(raw: Any) -> Dict[str, str]:
    result = _empty_slots()
    if isinstance(raw, dict):
        for k in PROMPT_SLOT_KEYS:
            if k in raw and raw[k] is not None:
                val = raw[k]
                if isinstance(val, list):
                    val = ", ".join(str(x).strip() for x in val if str(x).strip())
                result[k] = str(val).strip()
    return result


def _assemble_prompt_from_slots(slots: Dict[str, str], output_style: str = "auto") -> str:
    non_empty = [str(slots.get(k) or "").strip() for k in PROMPT_SLOT_KEYS if str(slots.get(k) or "").strip()]
    if not non_empty:
        return ""
    if output_style == "natural_language":
        return " ".join(v if v.endswith((".", "!", "?")) else f"{v}." for v in non_empty)
    return ", ".join(v.strip(", ") for v in non_empty)


def _format_slots_json(slots: Dict[str, str]) -> str:
    norm = _normalize_slots(slots)
    return json.dumps(norm, indent=2, ensure_ascii=False) if any(norm.values()) else ""


def _resolve_preserved_slots(preserve_slots: str) -> Tuple[str, ...]:
    return PRESERVE_SLOT_PRESETS.get(preserve_slots, ())


def _encode_pil_for_vlm(pil_img: Image.Image, max_side: int = 1024) -> str:
    vlm_img = prepare_rgb_image(pil_img).copy()
    if max(vlm_img.size) > max_side:
        vlm_img.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO()
    vlm_img.save(buf, format="JPEG", quality=90)
    jpeg_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{jpeg_b64}"


def _decode_image_data_url(data_url: str) -> Image.Image:
    raw_b64 = data_url.split(",", 1)[1] if "," in data_url else data_url
    img_bytes = base64.b64decode(raw_b64)
    return prepare_rgb_image(Image.open(io.BytesIO(img_bytes)))


def _tensor_to_pil_frames(tensor: Any, slot_name: str) -> List[Tuple[str, Image.Image]]:
    if tensor is None:
        return []
    batch_items = list(tensor) if (hasattr(tensor, "dim") and tensor.dim() == 4) else [tensor]
    out: List[Tuple[str, Image.Image]] = []
    for idx, frame in enumerate(batch_items, start=1):
        arr = (frame.detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        pil_img = prepare_rgb_image(Image.fromarray(arr))
        label = slot_name if len(batch_items) == 1 else f"{slot_name}[{idx}]"
        out.append((label, pil_img))
    return out


def _plan_image_groups(
    socket_tensors: List[Tuple[str, Any]],
    batch_mode: str = "auto",
) -> List[List[Tuple[str, Image.Image]]]:
    """Partition connected image sockets into execution groups based on batch_mode."""
    connected_slots: List[List[Tuple[str, Image.Image]]] = []
    for slot_name, tensor in socket_tensors:
        if tensor is None:
            continue
        frames = _tensor_to_pil_frames(tensor, slot_name)
        if frames:
            connected_slots.append(frames)

    if not connected_slots:
        return [[]]

    effective_mode = batch_mode
    if effective_mode == "auto":
        if len(connected_slots) == 1 and len(connected_slots[0]) > 1:
            effective_mode = "per_image"
        else:
            effective_mode = "combine_multi"

    if effective_mode == "per_image":
        groups: List[List[Tuple[str, Image.Image]]] = []
        for slot_frames in connected_slots:
            for item in slot_frames:
                groups.append([item])
        return groups

    combined: List[Tuple[str, Image.Image]] = []
    for slot_frames in connected_slots:
        for item in slot_frames:
            if len(combined) >= MAX_IMAGES:
                return [combined]
            combined.append(item)
    return [combined]


def _plan_preview_image_groups(
    pil_images: List[Tuple[str, Image.Image]],
    batch_mode: str = "auto",
) -> List[List[Tuple[str, Image.Image]]]:
    if not pil_images:
        return [[]]
    if batch_mode == "per_image":
        return [[item] for item in pil_images]
    return [pil_images[:MAX_IMAGES]]


def _extract_structured_output(
    raw_text: str,
    output_style: str = "auto",
    fallback_wd14: str = "",
) -> Tuple[str, Dict[str, str], str]:
    """Parse Stage 3 JSON (`summary`, `slots`, `final_prompt`) with resilient fallback."""
    cleaned, _ = extract_thinking_from_text(raw_text)

    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    candidate = fence_match.group(1) if fence_match else cleaned

    parsed_obj: Optional[Dict[str, Any]] = None
    for text in (candidate, cleaned):
        try:
            obj = json.loads(text)
            if isinstance(obj, dict) and any(k in obj for k in ("final_prompt", "slots", "summary", "chat_reply")):
                parsed_obj = obj
                break
        except json.JSONDecodeError:
            pass

    if parsed_obj is None:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start != -1 and end > start:
            try:
                obj = json.loads(candidate[start : end + 1])
                if isinstance(obj, dict):
                    parsed_obj = obj
            except json.JSONDecodeError:
                pass

    if parsed_obj is None:
        return "", _empty_slots(), cleaned or fallback_wd14

    summary = str(parsed_obj.get("summary") or parsed_obj.get("chat_reply") or "").strip()
    slots = _normalize_slots(parsed_obj.get("slots"))
    final_prompt = str(parsed_obj.get("final_prompt") or "").strip()
    if not final_prompt:
        final_prompt = _assemble_prompt_from_slots(slots, output_style=output_style) or cleaned or fallback_wd14
    return summary, slots, final_prompt


def _resolve_output_style_instructions(output_style: str) -> str:
    if output_style in OUTPUT_STYLE_SKILLS:
        return OUTPUT_STYLE_SKILLS[output_style]
    if output_style and output_style.startswith("skill:"):
        custom_text = kernel_manager.load_prompt_skill_text(output_style)
        if custom_text:
            return f"RUNTIME PROMPT SKILL [{output_style}]:\n{custom_text}"
    return OUTPUT_STYLE_SKILLS["auto"]


def _build_stage2_messages(
    vlm_images: List[Tuple[str, str]],
    wd14_items: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    count = len(vlm_images)
    anchor_blocks = [format_tiered_wd14_item(it) for it in wd14_items]
    anchor_text = (
        "Tiered objective Danbooru tags extracted individually per image by WD14 Tagger:\n"
        + "\n".join(anchor_blocks)
        + "\n\n"
        "WD14 vs VISION DIVISION OF LABOR:\n"
        "- Trust WD14 Character & Core Anchors (>=0.70) for factual character names, hair/eye colors, garments, and accessories.\n"
        "- Use your direct visual inspection to establish spatial layout, left/right positions, pose interaction, camera framing, "
        "lighting direction, atmosphere, and to correct any low-confidence WD14 false positives.\n\n"
        if anchor_blocks
        else ""
    )

    if count == 1:
        task_text = (
            f"{anchor_text}"
            "Analyze the image objectively and write a concise visual baseline in English covering all 7 dimensions: "
            "Subject/Character, Clothing/Accessories, Pose/Expression, Background/Setting, Lighting/Atmosphere, Composition/Camera, and Art Style."
        )
    else:
        task_text = (
            f"{anchor_text}"
            f"You are given {count} reference images labeled Image 1 through Image {count}. "
            "For EACH image individually ('Image 1:', 'Image 2:', ...), objectively summarize its Subject, Clothing, Pose/Expression, "
            "Background, Lighting, Composition, and Art Style grounded in its corresponding WD14 anchors while correcting tag noise. "
            "Then add a brief 'Cross-Image Summary:' noting complementary elements across images."
        )

    user_content: List[Dict[str, Any]] = [{"type": "text", "text": task_text}]
    for idx, (img_name, data_url) in enumerate(vlm_images, start=1):
        label = f"Image {idx} ({img_name}):" if img_name else f"Image {idx}:"
        user_content.append({"type": "text", "text": label})
        user_content.append({"type": "image_url", "image_url": {"url": data_url}})

    return [
        {
            "role": "system",
            "content": "You are a precise multi-image computer-vision captioning engine producing objective, spatially accurate visual baselines for ComfyUI prompt pipelines.",
        },
        {"role": "user", "content": user_content},
    ]


def _build_stage3_messages(
    wd14_tags: str,
    visual_baseline: str,
    transform_goal: str,
    preserve_slots: str,
    output_style: str,
    custom_system_prompt: Optional[str] = None,
) -> List[Dict[str, Any]]:
    context_sections: List[str] = []
    if wd14_tags:
        context_sections.append(f"[WD14 Per-Image Tiered Anchor Tags]\n{wd14_tags}")
    if visual_baseline:
        context_sections.append(f"[VLM Visual Baseline]\n{visual_baseline}")

    preserved = _resolve_preserved_slots(preserve_slots)
    if preserved:
        context_sections.append(
            "[PROTECTED DIMENSIONS — KEEP FAITHFUL TO ORIGINAL IMAGE & DO NOT LEAVE EMPTY]\n"
            + ", ".join(preserved)
        )

    context_blob = "\n\n".join(context_sections) if context_sections else "(No reference image attached — pure text prompt mode.)"
    style_block = _resolve_output_style_instructions(output_style)
    preset_block = f"\nUSER SYSTEM PROMPT PRESET:\n{custom_system_prompt}\n" if custom_system_prompt else ""

    system_prompt = (
        "You are an automated ComfyUI Prompt Transformation Agent.\n"
        "For every incoming image or multi-image group, apply the user's TRANSFORMATION GOAL cleanly and deterministically.\n\n"
        "CORE ENGINEERING RULES:\n"
        "1. WD14 vs VLM PRIORITY: Rely on WD14 Character & High-Confidence (>=0.70) tags for factual character identity, hair/eyes, "
        "and garment details; rely on the VLM Visual Baseline for spatial layout, pose, lighting direction, composition, and correcting low-confidence tag noise.\n"
        "2. MINIMAL INTRUSION & CONFLICT RESOLUTION: Modify ONLY the visual dimensions targeted by TRANSFORMATION GOAL "
        "(e.g. gender/presentation, character identity, clothing, pose, background, or art style). "
        "Whenever an attribute changes, actively DELETE every contradicting tag or phrase from the affected slot(s) and final_prompt "
        "(e.g. remove '1girl, breasts, cleavage' when transforming a female character into an otokonoko/male; remove old garments when swapping outfits), "
        "while keeping all unmentioned visual dimensions faithful to the source image.\n"
        "3. MULTI-IMAGE ROUTING: When multiple reference images (Image 1..N) are provided, assign each image's role (subject, outfit, pose, background, style) "
        "strictly according to TRANSFORMATION GOAL and eliminate cross-image contradictions.\n"
        "4. PROTECTED DIMENSIONS: Every dimension listed under PROTECTED DIMENSIONS must remain faithful to Image 1 (or the source image) and be populated in `slots`.\n"
        "5. CONTROLLED EXPANSION: Add only concrete, visible physical/photographic details; never pad with empty buzzwords.\n\n"
        f"{style_block}\n"
        f"{preset_block}\n"
        "STRICT JSON OUTPUT CONTRACT:\n"
        "Respond with ONE valid JSON object containing exactly these three keys:\n"
        '  - "summary": A short 1-sentence note (matching the user goal language) summarizing what was transformed and preserved.\n'
        '  - "slots": An object with all 7 string keys: "subject", "clothing", "pose_expression", "background", "lighting", "composition", "style" (strictly in English; "" if unused).\n'
        '  - "final_prompt": The complete, self-contained, ready-to-use English prompt synthesized from `slots` according to the active RUNTIME PROMPT SKILL. Do not wrap in markdown.\n\n'
        f"INPUT VISUAL CONTEXT:\n{context_blob}"
    )

    effective_goal = transform_goal.strip() or (
        "Extract an accurate, deduplicated, conflict-free high-fidelity prompt faithfully describing the input image(s) across all 7 visual slots."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"TRANSFORMATION GOAL:\n{effective_goal}"},
    ]


async def _run_pipeline_groups(
    groups: List[List[Tuple[str, Image.Image]]],
    llm_model_name: str,
    mmproj_model_name: str,
    output_style: str,
    preserve_slots: str,
    transform_goal: str,
    system_prompt_name: str,
    wd14_model_name: str,
    wd14_threshold: float,
    wd14_char_threshold: float,
    exclude_tags: str,
    enable_wd14: bool,
    ctx_size: int,
    memory_mode: str,
    n_gpu_layers: int,
    n_cpu_moe_layers: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repeat_penalty: float,
    seed: int,
    timeout_seconds: int,
    reasoning: str,
    extra_args: str,
    pbar: Any = None,
) -> List[Dict[str, Any]]:
    """Two-Phase Batch Scheduler:

    Phase 1: Load WD14 ONNX once, tag all image groups, and immediately free ONNX VRAM.
    Phase 2: Launch llama-server once, run VLM + Stage 3 rule transformation across all groups.
    """
    model_path = kernel_manager.resolve_model_path(llm_model_name, is_mmproj=False)
    if model_path is None:
        raise ValueError("Please select a valid GGUF model from ComfyUI/models/LLM.")
    mmproj_path = kernel_manager.resolve_model_path(mmproj_model_name, is_mmproj=True)
    custom_system_prompt = kernel_manager.load_system_prompt_text(system_prompt_name)

    # ------------------------------------------------------------------
    # Phase 1: Batch WD14 Tagging (single ONNX session load, zero overlap with llama-server VRAM)
    # ------------------------------------------------------------------
    wd14_group_results: List[Tuple[List[Dict[str, Any]], str]] = [([], "") for _ in groups]
    global_wd14_warning: Optional[str] = None
    if enable_wd14 and any(groups):
        try:
            wd14_group_results = await run_wd14_batch_groups(
                groups,
                model_name=wd14_model_name,
                threshold=wd14_threshold,
                character_threshold=wd14_char_threshold,
                exclude_tags=exclude_tags,
                max_images=MAX_IMAGES,
            )
        except Exception as wd_exc:
            if comfy is not None and isinstance(wd_exc, comfy.model_management.InterruptProcessingException):
                raise
            logger.warning("[PromptAgent] WD14 Phase 1 warning: %s", wd_exc)
            global_wd14_warning = f"WD14 warning: {wd_exc}"

    # ------------------------------------------------------------------
    # Phase 2: Decoupled llama-server VLM + Rule Transformation
    # ------------------------------------------------------------------
    await kernel_manager.ensure_running(
        model_path=model_path,
        mmproj_path=mmproj_path,
        ctx_size=ctx_size,
        memory_mode=memory_mode,
        n_gpu_layers=n_gpu_layers,
        n_cpu_moe_layers=n_cpu_moe_layers,
        reasoning=reasoning,
        extra_args=extra_args,
        startup_timeout=max(60.0, float(timeout_seconds)),
    )

    results: List[Dict[str, Any]] = []
    for idx, pil_group in enumerate(groups):
        if comfy is not None and comfy.model_management.processing_interrupted():
            kernel_manager.unload()
            comfy.model_management.throw_exception_if_processing_interrupted()

        wd14_items, wd14_formatted = wd14_group_results[idx]
        flat_wd14_fallback = ", ".join(it.get("tags", "") for it in wd14_items if it.get("tags"))
        stage_notes: List[str] = []
        if global_wd14_warning:
            stage_notes.append(global_wd14_warning)

        capped_group = pil_group[:MAX_IMAGES]
        if len(pil_group) > MAX_IMAGES:
            stage_notes.append(f"Image limit is {MAX_IMAGES}; extra images were ignored.")

        vlm_images = [(name, _encode_pil_for_vlm(img)) for name, img in capped_group]
        item_seed = seed if seed < 0 else (int(seed) + idx)
        visual_baseline = ""

        try:
            if vlm_images:
                if mmproj_path is not None:
                    stage2_messages = _build_stage2_messages(vlm_images, wd14_items)
                    vlm_out = await kernel_manager.chat_completions(
                        messages=stage2_messages,
                        response_format=None,
                        temperature=0.3,
                        top_p=top_p,
                        top_k=top_k,
                        repeat_penalty=repeat_penalty,
                        max_tokens=min(max_tokens, 1536),
                        seed=item_seed,
                        timeout_seconds=timeout_seconds,
                    )
                    visual_baseline = vlm_out["content"]
                else:
                    stage_notes.append(
                        f"No mmproj selected (text-only mode); using tiered WD14 anchors from {len(vlm_images)} image(s)."
                    )

            stage3_messages = _build_stage3_messages(
                wd14_tags=wd14_formatted,
                visual_baseline=visual_baseline,
                transform_goal=transform_goal,
                preserve_slots=preserve_slots,
                output_style=output_style,
                custom_system_prompt=custom_system_prompt,
            )
            agent_out = await kernel_manager.chat_completions(
                messages=stage3_messages,
                response_format={"type": "json_object"},
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repeat_penalty=repeat_penalty,
                max_tokens=max_tokens,
                seed=item_seed,
                timeout_seconds=timeout_seconds,
            )
            summary, slots, final_prompt = _extract_structured_output(
                agent_out["content"],
                output_style=output_style,
                fallback_wd14=flat_wd14_fallback,
            )
            reasoning_text = agent_out.get("reasoning", "")
            perf_text = agent_out.get("perf", "")
        except Exception as item_exc:
            if comfy is not None and isinstance(item_exc, comfy.model_management.InterruptProcessingException):
                raise
            if len(groups) == 1:
                raise
            logger.warning("[PromptAgent] Batch item %d fallback due to error: %s", idx + 1, item_exc)
            stage_notes.append(f"Item {idx + 1} LLM error fallback: {item_exc}")
            summary = "Fallback to WD14 tags due to LLM error."
            slots = _empty_slots()
            final_prompt = flat_wd14_fallback or transform_goal
            reasoning_text = ""
            perf_text = ""

        results.append(
            {
                "summary": summary,
                "final_prompt": final_prompt,
                "slots": slots,
                "structured_slots": _format_slots_json(slots),
                "wd14_items": wd14_items,
                "wd14_tags": wd14_formatted,
                "visual_baseline": visual_baseline,
                "reasoning": reasoning_text,
                "perf": perf_text,
                "stage_notes": stage_notes,
            }
        )
        if pbar is not None:
            pbar.update(1)

    return results


def _run_async_sync(coro):
    try:
        asyncio.get_running_loop()
        in_running_loop = True
    except RuntimeError:
        in_running_loop = False

    if in_running_loop:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    return asyncio.run(coro)


# ======================================================================
# PromptServer API Routes, Loopback Security Guard & Smart Pre-Queue Hook
# ======================================================================

_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost", "::ffff:127.0.0.1"}


def _is_loopback_request(request: web.Request) -> bool:
    remote = (request.remote or "").strip().lower()
    return remote in _LOOPBACK_HOSTS


def _register_server_routes() -> None:
    if PromptServer is None or getattr(PromptServer, "instance", None) is None:
        return

    routes = PromptServer.instance.routes

    # Smart pre-queue hook: only unload warm llama-server if the queued workflow has no PromptAgent node
    # (when PromptAgent runs, its own finally block unloads llama-server right before downstream KSampler)
    def _on_prompt_queue(json_data):
        if not (kernel_manager.auto_unload_on_queue and kernel_manager.is_running()):
            return json_data
        prompt_graph = (json_data or {}).get("prompt") if isinstance(json_data, dict) else None
        has_prompt_agent = False
        if isinstance(prompt_graph, dict):
            for node_def in prompt_graph.values():
                if isinstance(node_def, dict) and node_def.get("class_type") == "PromptAgent":
                    has_prompt_agent = True
                    break
        if not has_prompt_agent:
            logger.info("[PromptAgent] Queue Prompt (no PromptAgent in graph) — unloading warm llama-server.")
            kernel_manager.unload()
        return json_data

    PromptServer.instance.add_on_prompt_handler(_on_prompt_queue)

    @routes.get("/prompt_agent/status")
    async def prompt_agent_status(request):
        status = kernel_manager.get_status()
        status["local_version"] = await kernel_manager.get_local_version()
        return web.json_response({"ok": True, **status})

    @routes.get("/prompt_agent/models")
    async def prompt_agent_models(request):
        return web.json_response(
            {
                "ok": True,
                "llm_models": kernel_manager.list_llm_models(),
                "mmproj_models": kernel_manager.list_mmproj_models(),
                "output_styles": kernel_manager.list_output_styles(),
                "system_prompts": kernel_manager.list_system_prompts(),
                "wd14_models": list_wd14_models(),
            }
        )

    @routes.post("/prompt_agent/unload")
    async def prompt_agent_unload(request):
        was_running = kernel_manager.unload()
        gc.collect()
        return web.json_response(
            {
                "ok": True,
                "unloaded": was_running,
                "message": "llama-server stopped and VRAM released." if was_running else "llama-server was already idle.",
            }
        )

    @routes.get("/prompt_agent/check_update")
    async def prompt_agent_check_update(request):
        force = request.query.get("force", "").lower() in ("1", "true", "yes")
        return web.json_response(await kernel_manager.check_for_updates(force=force))

    @routes.post("/prompt_agent/update_kernel")
    async def prompt_agent_update_kernel(request):
        if not _is_loopback_request(request):
            return web.json_response({"ok": False, "error": "Kernel management is restricted to localhost."}, status=403)
        try:
            body = await request.json()
        except Exception:
            body = {}
        backend = str(body.get("backend", "auto"))
        try:
            return web.json_response(kernel_manager.start_update_task(backend=backend))
        except Exception as exc:
            logger.exception("[PromptAgent] Failed to start kernel update")
            return web.json_response({"ok": False, "error": str(exc)}, status=500)

    @routes.get("/prompt_agent/update_progress")
    async def prompt_agent_update_progress(request):
        return web.json_response({"ok": True, **kernel_manager.get_update_progress()})

    @routes.post("/prompt_agent/cancel_update")
    async def prompt_agent_cancel_update(request):
        if not _is_loopback_request(request):
            return web.json_response({"ok": False, "error": "Kernel management is restricted to localhost."}, status=403)
        return web.json_response(kernel_manager.cancel_update_task())

    @routes.post("/prompt_agent/skip_version")
    async def prompt_agent_skip_version(request):
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON body"}, status=400)
        version = str(body.get("version", "")).strip()
        if not version:
            return web.json_response({"ok": False, "error": "Missing version tag"}, status=400)
        return web.json_response(kernel_manager.skip_version(version))

    @routes.post("/prompt_agent/set_custom_binary")
    async def prompt_agent_set_custom_binary(request):
        if not _is_loopback_request(request):
            return web.json_response({"ok": False, "error": "Kernel management is restricted to localhost."}, status=403)
        try:
            body = await request.json()
        except Exception as exc:
            return web.json_response({"ok": False, "error": f"Invalid JSON: {exc}"}, status=400)
        raw_path = str(body.get("path") or "")
        try:
            return web.json_response(await kernel_manager.set_custom_binary_path(raw_path))
        except (FileNotFoundError, ValueError) as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        except Exception as exc:
            logger.exception("[PromptAgent] Failed to set custom binary path")
            return web.json_response({"ok": False, "error": str(exc)}, status=500)

    @routes.post("/prompt_agent/install_local_archive")
    async def prompt_agent_install_local_archive(request):
        if not _is_loopback_request(request):
            return web.json_response({"ok": False, "error": "Kernel management is restricted to localhost."}, status=403)
        try:
            ctype = (request.content_type or "").lower()
            if "multipart/" in ctype:
                reader = await request.multipart()
                files_data: List[Tuple[str, bytes]] = []
                while True:
                    part = await reader.next()
                    if part is None:
                        break
                    if part.filename:
                        files_data.append((part.filename, await part.read(decode=False)))
                return web.json_response(await kernel_manager.install_from_local_archives(files_data))

            body = await request.json()
            raw_paths = body.get("paths")
            if isinstance(raw_paths, str):
                raw_paths = [line.strip() for line in raw_paths.splitlines() if line.strip()]
            elif not isinstance(raw_paths, list):
                raw_paths = []
            return web.json_response(await kernel_manager.install_from_local_paths([str(p) for p in raw_paths]))
        except (FileNotFoundError, ValueError) as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        except Exception as exc:
            logger.exception("[PromptAgent] Manual archive installation failed")
            return web.json_response({"ok": False, "error": str(exc)}, status=500)

    @routes.post("/prompt_agent/preview")
    async def prompt_agent_preview(request):
        """Dry-run only this Prompt Agent node against connected images without triggering downstream sampling."""
        try:
            body = await request.json()
        except Exception as exc:
            return web.json_response({"ok": False, "error": f"Invalid JSON payload: {exc}"}, status=400)

        auto_unload = bool(body.get("auto_unload_vram", True))
        kernel_manager.auto_unload_on_queue = auto_unload

        pil_images: List[Tuple[str, Image.Image]] = []
        if isinstance(body.get("images"), list):
            try:
                for idx, item in enumerate(body["images"][:MAX_IMAGES], start=1):
                    if isinstance(item, dict) and item.get("data_url"):
                        name = str(item.get("name") or f"image_{idx}")
                        pil_images.append((name, _decode_image_data_url(str(item["data_url"]))))
                    elif isinstance(item, str) and item.strip():
                        pil_images.append((f"image_{idx}", _decode_image_data_url(item)))
            except Exception as img_exc:
                return web.json_response({"ok": False, "error": f"Failed to decode image: {img_exc}"}, status=400)

        transform_goal = str(body.get("transform_goal") or "").strip()
        if not pil_images and not transform_goal:
            return web.json_response(
                {"ok": False, "error": "Connect at least one image (image_1..6) or write a transform_goal first."},
                status=400,
            )

        batch_mode = str(body.get("batch_mode") or "auto")
        groups = _plan_preview_image_groups(pil_images, batch_mode=batch_mode)

        try:
            items = await _run_pipeline_groups(
                groups=groups,
                llm_model_name=str(body.get("llm_model") or ""),
                mmproj_model_name=str(body.get("mmproj_model") or "none"),
                output_style=str(body.get("output_style") or "auto"),
                preserve_slots=str(body.get("preserve_slots") or "none"),
                transform_goal=transform_goal,
                system_prompt_name=str(body.get("system_prompt") or "none"),
                wd14_model_name=str(body.get("wd14_model") or DEFAULT_WD14_MODEL),
                wd14_threshold=float(body.get("wd14_threshold", 0.35)),
                wd14_char_threshold=float(body.get("wd14_character_threshold", 0.85)),
                exclude_tags=str(body.get("exclude_tags", DEFAULT_WD14_EXCLUDE_TAGS)),
                enable_wd14=bool(body.get("enable_wd14", True)),
                ctx_size=int(body.get("ctx_size", 8192)),
                memory_mode=str(body.get("memory_mode") or "auto"),
                n_gpu_layers=int(body.get("n_gpu_layers", 99)),
                n_cpu_moe_layers=int(body.get("n_cpu_moe_layers", 1)),
                max_tokens=int(body.get("max_tokens", 2048)),
                temperature=float(body.get("temperature", 0.7)),
                top_p=float(body.get("top_p", 0.8)),
                top_k=int(body.get("top_k", 20)),
                repeat_penalty=float(body.get("repeat_penalty", 1.0)),
                seed=int(body.get("seed", 1)),
                timeout_seconds=int(body.get("timeout_seconds", 300)),
                reasoning=str(body.get("reasoning") or "off"),
                extra_args=str(body.get("extra_args") or ""),
            )
            status = kernel_manager.get_status()
            return web.json_response(
                {
                    "ok": True,
                    "items": items,
                    "model_loaded": status.get("model"),
                    "mmproj_loaded": status.get("mmproj"),
                }
            )
        except (ValueError, FileNotFoundError) as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        except Exception as exc:
            logger.exception("[PromptAgent] Preview pipeline error")
            return web.json_response({"ok": False, "error": str(exc)}, status=500)
        finally:
            if auto_unload and kernel_manager.is_running():
                kernel_manager.unload()
                gc.collect()


_register_server_routes()


# ======================================================================
# Node 1: PromptAgent (Plan A Static Rules + Dynamic Dataflow Pipeline)
# ======================================================================

class PromptAgent:
    """Static-rule / dynamic-dataflow prompt transformation node powered by llama-server & built-in WD14."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_model": (kernel_manager.list_llm_models(), {
                    "tooltip": "GGUF model loaded from ComfyUI/models/LLM (mmproj files are listed separately).",
                }),
                "mmproj_model": (kernel_manager.list_mmproj_models(), {
                    "default": "none",
                    "tooltip": "Vision projector GGUF for VLM Stage 2. Leave as 'none' for text-only + WD14 mode.",
                }),
                "output_style": (kernel_manager.list_output_styles(), {
                    "default": "auto",
                    "tooltip": "Runtime Prompt Skill: auto, natural_language (Flux/Qwen), danbooru_tags (Illustrious/SDXL), hybrid, edit_instruction (Flux Kontext/Qwen-Edit), structured_slots, custom_preset, or prompt_skills/*.",
                }),
                "preserve_slots": (list(PRESERVE_SLOT_PRESETS.keys()), {
                    "default": "none",
                    "tooltip": "Visual dimensions explicitly locked to the original image.",
                }),
                "transform_goal": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "dynamicPrompts": True,
                        "placeholder": "Persistent transformation rule applied to every incoming image (leave empty for high-fidelity reverse caption)...",
                        "tooltip": "Static transformation instruction saved with the workflow and applied dynamically to every incoming image/batch.",
                    },
                ),
                "batch_mode": (["auto", "per_image", "combine_multi"], {
                    "default": "auto",
                    "tooltip": "auto: single-socket multi-frame batch runs per_image; multiple connected sockets (image_1..6) run combine_multi.",
                }),
                "wd14_model": (list_wd14_models(), {
                    "default": DEFAULT_WD14_MODEL,
                    "tooltip": "Built-in self-contained WD14 ONNX tagger (loads from models/wd14/ or ComfyUI/models/wd14_tagger/).",
                }),
                "system_prompt": (kernel_manager.list_system_prompts(), {
                    "default": "none",
                    "tooltip": "Optional custom system prompt preset from ComfyUI/models/LLM/prompts/*.txt.",
                }),
            },
            "optional": {
                "image_1": ("IMAGE", {"tooltip": "Primary image or batch input #1 (single-socket batch auto-runs per image)."}),
                "image_2": ("IMAGE", {"tooltip": "Reference image #2 (e.g. target character, outfit, pose, or style)."}),
                "image_3": ("IMAGE", {"tooltip": "Reference image #3."}),
                "image_4": ("IMAGE", {"tooltip": "Reference image #4."}),
                "image_5": ("IMAGE", {"tooltip": "Reference image #5."}),
                "image_6": ("IMAGE", {"tooltip": "Reference image #6."}),
                "goal_override": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Optional dynamic upstream text overriding transform_goal.",
                }),
                "auto_unload_vram": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Terminate llama-server immediately after this node finishes so downstream Flux/SDXL gets 100% VRAM.",
                    "advanced": True,
                }),
                "enable_wd14": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Run Phase 1 tiered WD14 tagger per image before VLM/LLM.",
                    "advanced": True,
                }),
                "wd14_threshold": ("FLOAT", {
                    "default": 0.35, "min": 0.0, "max": 1.0, "step": 0.05,
                    "advanced": True,
                }),
                "wd14_character_threshold": ("FLOAT", {
                    "default": 0.85, "min": 0.0, "max": 1.0, "step": 0.05,
                    "advanced": True,
                }),
                "exclude_tags": ("STRING", {
                    "default": DEFAULT_WD14_EXCLUDE_TAGS,
                    "multiline": False,
                    "tooltip": "Comma-separated WD14 watermark/metadata tags to filter out.",
                    "advanced": True,
                }),
                "max_tokens": ("INT", {
                    "default": 2048, "min": 1, "max": 32768,
                    "advanced": True,
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05,
                    "advanced": True,
                }),
                "top_p": ("FLOAT", {
                    "default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01,
                    "advanced": True,
                }),
                "top_k": ("INT", {
                    "default": 20, "min": 1, "max": 1000,
                    "advanced": True,
                }),
                "repeat_penalty": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01,
                    "advanced": True,
                }),
                "ctx_size": ("INT", {
                    "default": 8192, "min": 1024, "max": 1048576, "step": 512,
                    "advanced": True,
                }),
                "seed": ("INT", {
                    "default": 1, "min": -1, "max": MAX_LLAMA_SEED,
                    "tooltip": "Fixed seed (>=0) enables ComfyUI native execution cache; -1 forces fresh generation every run.",
                    "advanced": True,
                }),
                "timeout_seconds": ("INT", {
                    "default": 300, "min": 10, "max": 3600,
                    "advanced": True,
                }),
                "reasoning": (["auto", "on", "off"], {
                    "default": "off",
                    "advanced": True,
                }),
                "memory_mode": ([
                    "auto",
                    "gpu_layers",
                    "cpu_moe_layers",
                    "gpu_and_cpu_moe_layers",
                ], {
                    "default": "auto",
                    "advanced": True,
                }),
                "n_gpu_layers": ("INT", {
                    "default": 99, "min": -1, "max": 999,
                    "advanced": True,
                }),
                "n_cpu_moe_layers": ("INT", {
                    "default": 1, "min": 1, "max": 999,
                    "advanced": True,
                }),
                "extra_args": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Optional safe llama-server CLI flags (e.g. --cache-type-k q8_0).",
                    "advanced": True,
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("final_prompt", "structured_slots", "wd14_tags", "reasoning", "perf")
    OUTPUT_IS_LIST = (True, True, True, True, True)
    OUTPUT_TOOLTIPS = (
        "Dynamic transformed English prompt(s) ready for CLIPTextEncode or image-editing models.",
        "JSON object(s) of the 7 structured visual slots.",
        "Per-image tiered WD14 objective anchor tags.",
        "Extracted model reasoning/thinking block when present.",
        "llama-server prompt & generation token throughput.",
    )
    FUNCTION = "execute"
    CATEGORY = "Prompt Agent"
    TITLE = "Prompt Agent (llama.cpp + WD14)"

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    @classmethod
    def IS_CHANGED(cls, seed=1, **kwargs):
        if int(seed) < 0:
            return math.nan
        return int(seed)

    def execute(
        self,
        llm_model,
        mmproj_model,
        output_style,
        preserve_slots,
        transform_goal,
        batch_mode,
        wd14_model,
        system_prompt,
        image_1=None,
        image_2=None,
        image_3=None,
        image_4=None,
        image_5=None,
        image_6=None,
        goal_override=None,
        auto_unload_vram=True,
        enable_wd14=True,
        wd14_threshold=0.35,
        wd14_character_threshold=0.85,
        exclude_tags=DEFAULT_WD14_EXCLUDE_TAGS,
        max_tokens=2048,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        repeat_penalty=1.0,
        ctx_size=8192,
        seed=1,
        timeout_seconds=300,
        reasoning="off",
        memory_mode="auto",
        n_gpu_layers=99,
        n_cpu_moe_layers=1,
        extra_args="",
    ):
        kernel_manager.auto_unload_on_queue = bool(auto_unload_vram)
        effective_goal = (
            goal_override.strip()
            if isinstance(goal_override, str) and goal_override.strip()
            else (transform_goal or "")
        )

        groups = _plan_image_groups(
            [
                ("image_1", image_1),
                ("image_2", image_2),
                ("image_3", image_3),
                ("image_4", image_4),
                ("image_5", image_5),
                ("image_6", image_6),
            ],
            batch_mode=batch_mode,
        )

        pbar = comfy.utils.ProgressBar(len(groups)) if comfy is not None else None
        try:
            preview_items = _run_async_sync(
                _run_pipeline_groups(
                    groups=groups,
                    llm_model_name=llm_model,
                    mmproj_model_name=mmproj_model,
                    output_style=output_style,
                    preserve_slots=preserve_slots,
                    transform_goal=effective_goal,
                    system_prompt_name=system_prompt,
                    wd14_model_name=wd14_model,
                    wd14_threshold=wd14_threshold,
                    wd14_char_threshold=wd14_character_threshold,
                    exclude_tags=exclude_tags,
                    enable_wd14=enable_wd14,
                    ctx_size=ctx_size,
                    memory_mode=memory_mode,
                    n_gpu_layers=n_gpu_layers,
                    n_cpu_moe_layers=n_cpu_moe_layers,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repeat_penalty=repeat_penalty,
                    seed=seed,
                    timeout_seconds=timeout_seconds,
                    reasoning=reasoning,
                    extra_args=extra_args,
                    pbar=pbar,
                )
            )
        finally:
            if auto_unload_vram and kernel_manager.is_running():
                kernel_manager.unload()
                gc.collect()

        final_prompts = [it["final_prompt"] for it in preview_items]
        slots_list = [it["structured_slots"] for it in preview_items]
        wd14_list = [it["wd14_tags"] for it in preview_items]
        reasonings = [it["reasoning"] for it in preview_items]
        perfs = [it["perf"] for it in preview_items]

        return {
            "ui": {"preview": preview_items},
            "result": (final_prompts, slots_list, wd14_list, reasonings, perfs),
        }


# ======================================================================
# Node 2: Universal Text & Reasoning Preview Node
# ======================================================================

def _flatten_str_input(val: Any) -> List[str]:
    if val is None:
        return []
    if isinstance(val, (list, tuple)):
        out: List[str] = []
        for item in val:
            if isinstance(item, (list, tuple)):
                out.extend(str(sub) for sub in item if sub is not None)
            elif item is not None:
                out.append(str(item))
        return out
    return [str(val)]


class PromptAgentTextPreview:
    """Universal text & reasoning preview node compatible with any LLM/VLM/Prompt node in ComfyUI.

    Accepts any STRING (single or batch list), automatically separates inline thinking tags
    (<think>...</think>, <reasoning>...</reasoning>, [Start thinking]...[End thinking], etc.)
    or reads an explicit `reasoning_in` input, displays both in a tabbed/collapsible UI,
    and forwards `clean_text` (thinking stripped), `thinking`, and `raw_text` downstream.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "text": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Any text, prompt, JSON, or raw LLM output (automatically extracts inline <think> blocks if reasoning_in is not connected).",
                }),
                "reasoning_in": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Optional dedicated reasoning/thinking input port (e.g. from Prompt Agent or LLM Text Processor).",
                }),
                "meta_in": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Optional metadata, performance speed, or WD14 tags to display alongside the preview.",
                }),
                "strip_thinking_from_output": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "When enabled, `clean_text` strips <think>...</think> blocks so downstream CLIPTextEncode receives only the final prompt.",
                }),
            },
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("clean_text", "thinking", "raw_text")
    OUTPUT_IS_LIST = (True, True, True)
    OUTPUT_NODE = True
    OUTPUT_TOOLTIPS = (
        "Clean text with thinking/reasoning blocks removed (safe to connect directly to CLIPTextEncode).",
        "Extracted or connected reasoning/thinking content.",
        "Original unmodified input text.",
    )
    FUNCTION = "preview_text"
    CATEGORY = "Prompt Agent"
    TITLE = "Text & Reasoning Preview (Universal)"

    def preview_text(
        self,
        text=None,
        reasoning_in=None,
        meta_in=None,
        strip_thinking_from_output=None,
    ):
        texts = _flatten_str_input(text)
        reasonings = _flatten_str_input(reasoning_in)
        metas = _flatten_str_input(meta_in)
        strip_list = _flatten_str_input(strip_thinking_from_output)
        should_strip = True if not strip_list else (str(strip_list[0]).lower() not in ("false", "0", ""))

        count = max(len(texts), len(reasonings), 1)
        clean_out: List[str] = []
        think_out: List[str] = []
        raw_out: List[str] = []
        ui_items: List[Dict[str, Any]] = []

        for idx in range(count):
            raw_item = texts[idx] if idx < len(texts) else ""
            explicit_think = reasonings[idx].strip() if idx < len(reasonings) else ""
            meta_item = metas[idx].strip() if idx < len(metas) else (metas[-1].strip() if metas else "")

            stripped_text, inline_think = extract_thinking_from_text(raw_item)
            combined_think = (
                f"{explicit_think}\n\n{inline_think}".strip()
                if (explicit_think and inline_think and explicit_think != inline_think)
                else (explicit_think or inline_think)
            )
            final_clean = stripped_text if should_strip else raw_item

            clean_out.append(final_clean)
            think_out.append(combined_think)
            raw_out.append(raw_item)
            ui_items.append(
                {
                    "index": idx + 1,
                    "clean_text": final_clean,
                    "thinking": combined_think,
                    "raw_text": raw_item,
                    "meta": meta_item,
                }
            )

        return {
            "ui": {"items": ui_items},
            "result": (clean_out, think_out, raw_out),
        }


NODE_CLASS_MAPPINGS = {
    "PromptAgent": PromptAgent,
    "PromptAgentTextPreview": PromptAgentTextPreview,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptAgent": "Prompt Agent (llama.cpp + WD14)",
    "PromptAgentTextPreview": "Text & Reasoning Preview (Universal)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
