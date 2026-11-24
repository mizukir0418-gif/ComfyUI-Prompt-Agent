import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const STYLE_ID = "comfyui-prompt-agent-plan-a-styles";
const MODAL_ID = "comfyui-prompt-agent-update-modal";
const MAX_IMAGES = 6;

const SLOT_DEFS = [
    { key: "subject", label: "Subject 主体" },
    { key: "clothing", label: "Clothing 服装" },
    { key: "pose_expression", label: "Pose 姿势表情" },
    { key: "background", label: "Background 背景" },
    { key: "lighting", label: "Lighting 光影" },
    { key: "composition", label: "Composition 构图" },
    { key: "style", label: "Style 画风" },
];

let latestUpdateInfo = null;
const kernelBadgeListeners = new Set();
let activeUpdatePollTimer = null;

function injectStyles() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
        .pa-panel {
            display: flex;
            flex-direction: column;
            width: 100%;
            background: #14171f;
            color: #e2e8f0;
            border: 1px solid #2a3140;
            border-radius: 9px;
            overflow: hidden;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            font-size: 12px;
            box-sizing: border-box;
        }
        .pa-panel * { box-sizing: border-box; }

        /* Compact 1-Line Native Toolbar */
        .pa-toolbar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 6px;
            padding: 6px 8px;
            background: #1a1f2b;
            border-bottom: 1px solid #283040;
            flex-wrap: wrap;
        }
        .pa-status {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            font-size: 11px;
            color: #94a3b8;
            min-width: 0;
            max-width: 210px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .pa-dot {
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: #64748b;
            flex-shrink: 0;
        }
        .pa-dot.running { background: #34d399; box-shadow: 0 0 6px rgba(52, 211, 153, 0.7); }
        .pa-dot.busy { background: #60a5fa; animation: pa-pulse 1s infinite; }
        .pa-dot.error { background: #ef4444; }

        .pa-toolbar-btns {
            display: flex;
            align-items: center;
            gap: 4px;
            flex-shrink: 0;
        }
        .pa-btn {
            background: #232a38;
            color: #d6dce8;
            border: 1px solid #343e52;
            border-radius: 6px;
            padding: 3px 8px;
            font-size: 11px;
            cursor: pointer;
            position: relative;
            white-space: nowrap;
            transition: background 0.15s, border-color 0.15s;
        }
        .pa-btn:hover { background: #2d3648; border-color: #4b5873; color: #fff; }
        .pa-btn:disabled { opacity: 0.45; cursor: not-allowed; }
        .pa-btn.primary {
            background: rgba(59, 130, 246, 0.22);
            border-color: #3b82f6;
            color: #93c5fd;
            font-weight: 600;
        }
        .pa-btn.primary:hover { background: rgba(59, 130, 246, 0.35); color: #fff; }
        .pa-btn.danger:hover { background: #3b2026; border-color: #ef4444; color: #fecaca; }
        .pa-btn.has-update::after {
            content: "";
            position: absolute;
            top: -3px;
            right: -3px;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #f59e0b;
            border: 1.5px solid #1a1f2b;
        }

        /* Compact Result Echo Area */
        .pa-echo {
            padding: 7px 8px;
            display: flex;
            flex-direction: column;
            gap: 6px;
        }
        .pa-echo-placeholder {
            color: #7c879c;
            font-size: 11.5px;
            line-height: 1.45;
            padding: 6px 4px;
            text-align: center;
        }
        .pa-echo-top {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 6px;
            font-size: 11px;
            color: #94a3b8;
        }
        .pa-pager {
            display: inline-flex;
            align-items: center;
            gap: 4px;
        }
        .pa-summary-line {
            font-size: 11.5px;
            color: #93c5fd;
            line-height: 1.4;
        }
        .pa-note-line {
            font-size: 11px;
            color: #fbbf24;
            line-height: 1.35;
        }
        .pa-err-box {
            padding: 6px 8px;
            border-radius: 6px;
            background: #2b181d;
            border: 1px solid #7f1d1d;
            color: #fecaca;
            font-size: 11.5px;
            word-break: break-word;
            white-space: pre-wrap;
        }
        .pa-prompt-box {
            background: #0e1118;
            border: 1px solid #262e3e;
            border-radius: 6px;
            padding: 7px 8px;
            max-height: 115px;
            overflow-y: auto;
            font-size: 11.5px;
            line-height: 1.48;
            color: #f1f5f9;
            white-space: pre-wrap;
            word-break: break-word;
            user-select: text;
        }
        .pa-details {
            background: #10141c;
            border: 1px solid #232a38;
            border-radius: 6px;
            padding: 4px 7px;
            font-size: 11px;
        }
        .pa-details summary {
            cursor: pointer;
            color: #a5b4fc;
            font-weight: 500;
            user-select: none;
        }
        .pa-details-body {
            margin-top: 5px;
            max-height: 130px;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 4px;
            color: #cbd5e1;
            line-height: 1.4;
            word-break: break-word;
            user-select: text;
        }
        .pa-slot-line {
            padding: 3px 5px;
            background: #151a24;
            border-radius: 4px;
        }
        .pa-slot-key {
            color: #93c5fd;
            font-weight: 600;
            margin-right: 5px;
        }
        .pa-tier-badge {
            display: inline-block;
            font-size: 9.5px;
            font-weight: 700;
            padding: 0 5px;
            border-radius: 4px;
            margin-right: 4px;
        }
        .pa-tier-badge.char { background: rgba(192, 132, 252, 0.2); color: #d8b4fe; border: 1px solid rgba(192, 132, 252, 0.4); }
        .pa-tier-badge.high { background: rgba(52, 211, 153, 0.18); color: #6ee7b7; border: 1px solid rgba(52, 211, 153, 0.4); }
        .pa-tier-badge.med { background: rgba(148, 163, 184, 0.16); color: #cbd5e1; border: 1px solid rgba(148, 163, 184, 0.35); }
        .pa-perf {
            font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
            font-size: 10.5px;
            color: #6ee7b7;
        }

        /* Kernel Update & Manual Install Modal */
        .pa-modal-backdrop {
            position: fixed;
            inset: 0;
            background: rgba(8, 10, 15, 0.72);
            backdrop-filter: blur(4px);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 10000;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }
        .pa-modal {
            width: min(580px, 92vw);
            max-height: 88vh;
            display: flex;
            flex-direction: column;
            background: #161a23;
            color: #e2e8f0;
            border: 1px solid #2f3748;
            border-radius: 14px;
            box-shadow: 0 24px 64px rgba(0, 0, 0, 0.55);
            overflow: hidden;
        }
        .pa-modal-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 13px 18px;
            background: linear-gradient(135deg, #1c2231, #211f36);
            border-bottom: 1px solid #2d3546;
        }
        .pa-modal-title {
            font-size: 15px;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .pa-sparkle {
            background: linear-gradient(135deg, #60a5fa, #c084fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 700;
        }
        .pa-modal-close {
            background: transparent;
            border: none;
            color: #94a3b8;
            font-size: 18px;
            cursor: pointer;
            padding: 2px 6px;
        }
        .pa-modal-close:hover { color: #fff; }
        .pa-modal-body {
            padding: 15px 18px;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 11px;
            font-size: 13px;
        }
        .pa-ver-row {
            display: flex;
            align-items: center;
            gap: 9px;
            flex-wrap: wrap;
        }
        .pa-ver-pill {
            padding: 4px 10px;
            border-radius: 999px;
            font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
            font-size: 12px;
            background: #232938;
            border: 1px solid #353f54;
        }
        .pa-ver-pill.latest {
            background: rgba(59, 130, 246, 0.16);
            border-color: #3b82f6;
            color: #93c5fd;
        }
        .pa-hw-box {
            padding: 8px 10px;
            background: rgba(30, 41, 59, 0.65);
            border: 1px solid #334155;
            border-radius: 8px;
            font-size: 12px;
            line-height: 1.45;
            color: #e2e8f0;
        }
        .pa-hw-reason {
            margin-top: 3px;
            font-size: 11.5px;
            color: #93c5fd;
        }
        .pa-notes-box {
            background: #10131a;
            border: 1px solid #262d3c;
            border-radius: 8px;
            padding: 10px 12px;
            max-height: 150px;
            overflow-y: auto;
            white-space: pre-wrap;
            font-size: 12px;
            line-height: 1.5;
            color: #cbd5e1;
        }
        .pa-backend-row {
            display: flex;
            align-items: center;
            gap: 10px;
            flex-wrap: wrap;
        }
        .pa-select {
            background: #10131a;
            color: #f1f5f9;
            border: 1px solid #353f54;
            border-radius: 7px;
            padding: 5px 10px;
            font-size: 12.5px;
        }
        .pa-asset-info {
            font-size: 11.5px;
            color: #94a3b8;
            word-break: break-all;
        }
        .pa-dl-links {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            align-items: center;
            font-size: 11.5px;
        }
        .pa-progress-wrap {
            display: none;
            flex-direction: column;
            gap: 5px;
        }
        .pa-progress-wrap.visible { display: flex; }
        .pa-progress-track {
            width: 100%;
            height: 8px;
            background: #10131a;
            border: 1px solid #2a3244;
            border-radius: 999px;
            overflow: hidden;
        }
        .pa-progress-fill {
            height: 100%;
            width: 0%;
            background: linear-gradient(90deg, #3b82f6, #8b5cf6);
            transition: width 0.25s ease;
        }
        .pa-progress-meta {
            display: flex;
            align-items: center;
            justify-content: space-between;
            font-size: 11px;
            color: #94a3b8;
            font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        }
        .pa-modal-status {
            font-size: 12.5px;
            padding: 8px 10px;
            border-radius: 8px;
            display: none;
        }
        .pa-modal-status.info { display: block; background: #17253d; color: #93c5fd; border: 1px solid #2563eb; }
        .pa-modal-status.success { display: block; background: #122c24; color: #6ee7b7; border: 1px solid #10b981; }
        .pa-modal-status.error { display: block; background: #2f171c; color: #fecaca; border: 1px solid #ef4444; }
        .pa-manual-panel {
            background: #10131a;
            border: 1px solid #283040;
            border-radius: 8px;
            padding: 8px 10px;
            font-size: 12px;
        }
        .pa-manual-panel summary {
            cursor: pointer;
            color: #c4b5fd;
            font-weight: 600;
            user-select: none;
        }
        .pa-manual-section {
            margin-top: 9px;
            padding-top: 8px;
            border-top: 1px solid #222938;
            display: flex;
            flex-direction: column;
            gap: 6px;
        }
        .pa-manual-label {
            font-size: 11.5px;
            font-weight: 600;
            color: #cbd5e1;
        }
        .pa-manual-row {
            display: flex;
            align-items: center;
            gap: 6px;
            flex-wrap: wrap;
        }
        .pa-manual-input {
            flex: 1;
            min-width: 180px;
            background: #161b26;
            color: #f1f5f9;
            border: 1px solid #334155;
            border-radius: 6px;
            padding: 5px 8px;
            font-size: 11.5px;
            font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
            outline: none;
        }
        .pa-manual-input:focus { border-color: #60a5fa; }
        .pa-modal-footer {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 8px;
            padding: 12px 18px;
            background: #131720;
            border-top: 1px solid #262d3c;
            flex-wrap: wrap;
        }
        .pa-modal-footer-right {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .pa-btn-primary {
            background: linear-gradient(135deg, #2563eb, #6366f1);
            color: #fff;
            border: none;
            border-radius: 8px;
            padding: 7px 14px;
            font-size: 12.5px;
            font-weight: 600;
            cursor: pointer;
        }
        .pa-btn-primary:hover { filter: brightness(1.1); }
        .pa-btn-primary:disabled { opacity: 0.45; cursor: not-allowed; }
        .pa-btn-secondary {
            background: #232938;
            color: #cbd5e1;
            border: 1px solid #353f54;
            border-radius: 8px;
            padding: 6px 12px;
            font-size: 12.5px;
            cursor: pointer;
        }
        .pa-btn-secondary:hover { background: #2d3548; color: #fff; }
        .pa-link {
            color: #60a5fa;
            text-decoration: none;
            font-size: 12px;
        }
        .pa-link:hover { text-decoration: underline; }

        @keyframes pa-pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.35; }
        }
    `;
    document.head.appendChild(style);
}

function formatBytes(bytes) {
    if (!bytes || bytes <= 0) return "";
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function buildApiUrl(path) {
    return typeof api.apiURL === "function" ? api.apiURL(path) : path;
}

// Native fetch without ComfyUI api.fetchApi's short timeout wrapper
async function longFetch(path, options = {}) {
    return fetch(buildApiUrl(path), options);
}

function notifyKernelBadgeListeners(info) {
    latestUpdateInfo = info;
    for (const cb of kernelBadgeListeners) {
        try { cb(info); } catch (_) {}
    }
}

function stopCanvasPropagation(el) {
    for (const ev of ["wheel", "pointerdown", "mousedown", "keydown", "keyup", "keypress"]) {
        el.addEventListener(ev, (e) => e.stopPropagation());
    }
}

function blobToDataUrl(blob) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(typeof reader.result === "string" ? reader.result : "");
        reader.onerror = () => reject(new Error("Failed to read image blob."));
        reader.readAsDataURL(blob);
    });
}

// ======================================================================
// Connected Socket Image Extractor (for single-node Preview dry run)
// ======================================================================

async function fetchImageFromUpstreamNode(originNode, slotName) {
    if (!originNode) return null;

    const imgWidget = originNode.widgets?.find(
        (w) => w.name === "image" && typeof w.value === "string" && w.value.trim()
    );
    if (imgWidget) {
        let raw = imgWidget.value.trim();
        let type = "input";
        const typeMatch = raw.match(/\s+\[(input|output|temp)\]$/i);
        if (typeMatch) {
            type = typeMatch[1].toLowerCase();
            raw = raw.slice(0, typeMatch.index).trim();
        }
        const norm = raw.replace(/\\/g, "/");
        const lastSlash = norm.lastIndexOf("/");
        const subfolder = lastSlash >= 0 ? norm.slice(0, lastSlash) : "";
        const filename = lastSlash >= 0 ? norm.slice(lastSlash + 1) : norm;

        const params = new URLSearchParams({ filename, type });
        if (subfolder) params.set("subfolder", subfolder);
        const resp = await longFetch(`/view?${params.toString()}`);
        if (resp.ok) {
            const dataUrl = await blobToDataUrl(await resp.blob());
            if (dataUrl) return { name: `${slotName}:${filename}`, data_url: dataUrl };
        }
    }

    const previewImg = originNode.imgs?.[0];
    if (previewImg && previewImg.src) {
        const resp = await fetch(previewImg.src);
        if (resp.ok) {
            const dataUrl = await blobToDataUrl(await resp.blob());
            if (dataUrl) return { name: `${slotName}:preview.png`, data_url: dataUrl };
        }
    }
    return null;
}

async function pullConnectedSocketImages(node) {
    if (!Array.isArray(node.inputs) || !app.graph?.links) return [];
    const pulled = [];
    for (let i = 1; i <= MAX_IMAGES; i++) {
        const slotName = `image_${i}`;
        const inp = node.inputs.find((item) => item.name === slotName);
        if (!inp || inp.link == null) continue;
        const link = app.graph.links[inp.link];
        if (!link) continue;
        const originNode = app.graph.getNodeById(link.origin_id);
        try {
            const item = await fetchImageFromUpstreamNode(originNode, slotName);
            if (item) pulled.push(item);
        } catch (err) {
            console.warn(`[PromptAgent] Failed reading ${slotName}:`, err);
        }
    }
    return pulled;
}

// ======================================================================
// Open-Source GPU-Aware Kernel Update & Manual Install Modal
// ======================================================================

function stopUpdatePolling() {
    if (activeUpdatePollTimer !== null) {
        clearInterval(activeUpdatePollTimer);
        activeUpdatePollTimer = null;
    }
}

function closeUpdateModal() {
    stopUpdatePolling();
    const existing = document.getElementById(MODAL_ID);
    if (existing) existing.remove();
}

function showKernelUpdateModal(info) {
    injectStyles();
    closeUpdateModal();

    const backdrop = document.createElement("div");
    backdrop.id = MODAL_ID;
    backdrop.className = "pa-modal-backdrop";

    const modal = document.createElement("div");
    modal.className = "pa-modal";

    const isInstalled = Boolean(info.installed);
    const hasUpdate = Boolean(info.update_available);
    const latestDisplay = info.latest_display || info.latest_version || "unknown";
    const titleText = !isInstalled
        ? "Install llama.cpp Kernel (llama-server)"
        : hasUpdate
            ? "llama.cpp Kernel Update Available"
            : "llama.cpp Kernel Manager";

    const header = document.createElement("div");
    header.className = "pa-modal-header";
    header.innerHTML = `<div class="pa-modal-title"><span class="pa-sparkle">✦</span><span>${titleText}</span></div>`;
    const closeBtn = document.createElement("button");
    closeBtn.className = "pa-modal-close";
    closeBtn.textContent = "×";
    closeBtn.onclick = closeUpdateModal;
    header.appendChild(closeBtn);

    const body = document.createElement("div");
    body.className = "pa-modal-body";

    const verRow = document.createElement("div");
    verRow.className = "pa-ver-row";

    const curPill = document.createElement("span");
    curPill.className = "pa-ver-pill";
    curPill.appendChild(document.createTextNode("Current: "));
    const curStrong = document.createElement("strong");
    curStrong.textContent = info.local_version || "not_installed";
    curPill.appendChild(curStrong);

    const arrowSpan = document.createElement("span");
    arrowSpan.textContent = "→";

    const latPill = document.createElement("span");
    latPill.className = "pa-ver-pill latest";
    latPill.appendChild(document.createTextNode("Latest: "));
    const latStrong = document.createElement("strong");
    latStrong.textContent = latestDisplay;
    latPill.appendChild(latStrong);

    verRow.appendChild(curPill);
    verRow.appendChild(arrowSpan);
    verRow.appendChild(latPill);
    const binPageUrl = info.binary_release_url || info.html_url || "https://github.com/ggml-org/llama.cpp/releases";
    const binLink = document.createElement("a");
    binLink.className = "pa-link";
    binLink.href = binPageUrl;
    binLink.target = "_blank";
    binLink.rel = "noopener noreferrer";
    binLink.textContent = info.binary_tag ? `Official Builds (${info.binary_tag}) ↗` : "Official Releases ↗";
    verRow.appendChild(binLink);

    const allRelUrl = info.official_releases_url || "https://github.com/ggml-org/llama.cpp/releases";
    if (allRelUrl !== binPageUrl) {
        const allLink = document.createElement("a");
        allLink.className = "pa-link";
        allLink.href = allRelUrl;
        allLink.target = "_blank";
        allLink.rel = "noopener noreferrer";
        allLink.textContent = "All Releases ↗";
        verRow.appendChild(allLink);
    }
    body.appendChild(verRow);

    if (info.hardware && info.hardware.summary) {
        const hwBox = document.createElement("div");
        hwBox.className = "pa-hw-box";
        const hwTitle = document.createElement("div");
        hwTitle.innerHTML = `<strong>Detected GPU:</strong> `;
        hwTitle.appendChild(document.createTextNode(info.hardware.summary));
        hwBox.appendChild(hwTitle);
        if (info.recommendation_reason) {
            const hwReason = document.createElement("div");
            hwReason.className = "pa-hw-reason";
            hwReason.textContent = `Auto-match: ${info.recommendation_reason}`;
            hwBox.appendChild(hwReason);
        }
        body.appendChild(hwBox);
    }

    if (info.binary_path) {
        const pathInfo = document.createElement("div");
        pathInfo.className = "pa-asset-info";
        const srcTag = info.binary_source ? ` [${info.binary_source}]` : "";
        pathInfo.textContent = `Active binary${srcTag}: ${info.binary_path}`;
        body.appendChild(pathInfo);
    }

    const notesBox = document.createElement("div");
    notesBox.className = "pa-notes-box";
    notesBox.textContent = info.release_notes || "No release notes provided for this release.";
    body.appendChild(notesBox);

    const backends = Array.isArray(info.available_backends) ? info.available_backends : [];
    const matchedAssets = info.matched_assets || {};
    let selectedBackend = info.recommended_backend || backends[0] || "auto";

    const backendRow = document.createElement("div");
    backendRow.className = "pa-backend-row";
    const backendLabel = document.createElement("label");
    backendLabel.textContent = "Target Backend:";
    const select = document.createElement("select");
    select.className = "pa-select";

    const assetDetail = document.createElement("div");
    assetDetail.className = "pa-dl-links";

    const refreshAssetDetail = () => {
        assetDetail.innerHTML = "";
        const meta = matchedAssets[select.value];
        if (meta && Array.isArray(meta.downloads) && meta.downloads.length > 0) {
            const prefix = document.createElement("span");
            prefix.className = "pa-asset-info";
            const sizeStr = formatBytes(meta.total_size);
            prefix.textContent = `Official package(s)${sizeStr ? ` (${sizeStr})` : ""}:`;
            assetDetail.appendChild(prefix);
            meta.downloads.forEach((dl, idx) => {
                if (idx > 0) {
                    const plus = document.createElement("span");
                    plus.className = "pa-asset-info";
                    plus.textContent = "+";
                    assetDetail.appendChild(plus);
                }
                const a = document.createElement("a");
                a.className = "pa-link";
                a.href = dl.url;
                a.target = "_blank";
                a.rel = "noopener noreferrer";
                a.textContent = `${dl.name} ↗`;
                assetDetail.appendChild(a);
            });
        } else {
            const span = document.createElement("span");
            span.className = "pa-asset-info";
            span.textContent = "Click 'Refresh Release' or open Official Releases manually.";
            assetDetail.appendChild(span);
        }
    };

    for (const bk of backends) {
        const opt = document.createElement("option");
        opt.value = bk;
        const lbl = matchedAssets[bk]?.label || bk.toUpperCase();
        opt.textContent = bk === info.recommended_backend ? `${lbl} (Recommended)` : lbl;
        if (bk === selectedBackend) opt.selected = true;
        select.appendChild(opt);
    }
    select.onchange = () => {
        selectedBackend = select.value;
        refreshAssetDetail();
    };
    refreshAssetDetail();

    if (backends.length > 0) {
        backendRow.appendChild(backendLabel);
        backendRow.appendChild(select);
        body.appendChild(backendRow);
    }
    body.appendChild(assetDetail);

    const progressWrap = document.createElement("div");
    progressWrap.className = "pa-progress-wrap";
    const progressTrack = document.createElement("div");
    progressTrack.className = "pa-progress-track";
    const progressFill = document.createElement("div");
    progressFill.className = "pa-progress-fill";
    progressTrack.appendChild(progressFill);
    const progressMeta = document.createElement("div");
    progressMeta.className = "pa-progress-meta";
    const progressLeft = document.createElement("span");
    const progressRight = document.createElement("span");
    progressMeta.appendChild(progressLeft);
    progressMeta.appendChild(progressRight);
    progressWrap.appendChild(progressTrack);
    progressWrap.appendChild(progressMeta);
    body.appendChild(progressWrap);

    const statusBox = document.createElement("div");
    statusBox.className = "pa-modal-status";
    body.appendChild(statusBox);

    const setStatus = (kind, msg) => {
        statusBox.className = `pa-modal-status ${kind}`;
        statusBox.textContent = msg;
    };

    // --- Collapsible Manual Download & Custom Kernel Panel ---
    const manualPanel = document.createElement("details");
    manualPanel.className = "pa-manual-panel";
    const manualSummary = document.createElement("summary");
    manualSummary.textContent = "Manual Download / Custom Kernel Path (手动下载与切换内核)";
    manualPanel.appendChild(manualSummary);

    const secArchive = document.createElement("div");
    secArchive.className = "pa-manual-section";
    secArchive.innerHTML = `<div class="pa-manual-label">1. Extract Manually Downloaded Official Archive(s) → bin/</div>`;

    const archivePathsInput = document.createElement("textarea");
    archivePathsInput.className = "pa-manual-input";
    archivePathsInput.rows = 2;
    archivePathsInput.placeholder = "Paste local .zip/.tar.gz path(s) (one per line) OR folder containing llama-*.zip & cudart-*.zip...";

    const archiveBtnRow = document.createElement("div");
    archiveBtnRow.className = "pa-manual-row";

    const openOfficialBtn = document.createElement("button");
    openOfficialBtn.className = "pa-btn";
    openOfficialBtn.textContent = "Open Official Page ↗";
    openOfficialBtn.onclick = () => window.open(binPageUrl, "_blank", "noopener,noreferrer");

    const extractLocalBtn = document.createElement("button");
    extractLocalBtn.className = "pa-btn primary";
    extractLocalBtn.textContent = "Extract & Install Path(s)";
    extractLocalBtn.onclick = async () => {
        const raw = archivePathsInput.value.trim();
        if (!raw) {
            setStatus("error", "Please paste a local .zip/.tar.gz file path or folder path first.");
            return;
        }
        extractLocalBtn.disabled = true;
        setStatus("info", "Extracting local archive(s) into bin/...");
        try {
            const resp = await longFetch("/prompt_agent/install_local_archive", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ paths: raw }),
            });
            const res = await resp.json();
            if (!resp.ok || !res.ok) throw new Error(res.error || `HTTP ${resp.status}`);
            setStatus("success", `Installed ${res.installed_version} into bin/ successfully!`);
            notifyKernelBadgeListeners({
                ...info,
                installed: true,
                binary_path: res.binary_path,
                binary_source: "local_bin",
                local_version: res.installed_version,
                update_available: false,
            });
        } catch (err) {
            setStatus("error", `Manual install failed: ${err.message}`);
        } finally {
            extractLocalBtn.disabled = false;
        }
    };

    const uploadZipInput = document.createElement("input");
    uploadZipInput.type = "file";
    uploadZipInput.accept = ".zip,.gz,.tgz";
    uploadZipInput.multiple = true;
    uploadZipInput.style.display = "none";
    uploadZipInput.onchange = async () => {
        if (!uploadZipInput.files?.length) return;
        const form = new FormData();
        for (const f of uploadZipInput.files) form.append("files", f, f.name);
        uploadZipInput.value = "";
        setStatus("info", "Uploading & extracting archive(s) into bin/...");
        try {
            const resp = await longFetch("/prompt_agent/install_local_archive", {
                method: "POST",
                body: form,
            });
            if (resp.status === 413) {
                throw new Error("Archive exceeds ComfyUI's 100MB HTTP upload limit. Paste the local .zip file path above instead (no size limit!).");
            }
            const res = await resp.json();
            if (!resp.ok || !res.ok) throw new Error(res.error || `HTTP ${resp.status}`);
            setStatus("success", `Installed ${res.installed_version} into bin/ successfully!`);
            notifyKernelBadgeListeners({
                ...info,
                installed: true,
                binary_path: res.binary_path,
                binary_source: "local_bin",
                local_version: res.installed_version,
                update_available: false,
            });
        } catch (err) {
            setStatus("error", `Upload install failed: ${err.message}`);
        }
    };

    const uploadZipBtn = document.createElement("button");
    uploadZipBtn.className = "pa-btn";
    uploadZipBtn.textContent = "Upload .zip (<100MB)";
    uploadZipBtn.onclick = () => uploadZipInput.click();

    archiveBtnRow.appendChild(openOfficialBtn);
    archiveBtnRow.appendChild(extractLocalBtn);
    archiveBtnRow.appendChild(uploadZipBtn);
    archiveBtnRow.appendChild(uploadZipInput);
    secArchive.appendChild(archivePathsInput);
    secArchive.appendChild(archiveBtnRow);
    manualPanel.appendChild(secArchive);

    const secCustom = document.createElement("div");
    secCustom.className = "pa-manual-section";
    secCustom.innerHTML = `<div class="pa-manual-label">2. Use Custom llama-server Executable / Folder Path</div>`;

    const customRow = document.createElement("div");
    customRow.className = "pa-manual-row";
    const customInput = document.createElement("input");
    customInput.className = "pa-manual-input";
    customInput.type = "text";
    customInput.value = info.custom_binary_path || "";
    customInput.placeholder = "Path to llama-server.exe (or folder containing it)...";

    const applyCustomBtn = document.createElement("button");
    applyCustomBtn.className = "pa-btn primary";
    applyCustomBtn.textContent = "Apply Path";
    applyCustomBtn.onclick = async () => {
        applyCustomBtn.disabled = true;
        try {
            const resp = await longFetch("/prompt_agent/set_custom_binary", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ path: customInput.value.trim() }),
            });
            const res = await resp.json();
            if (!resp.ok || !res.ok) throw new Error(res.error || `HTTP ${resp.status}`);
            setStatus("success", `Switched kernel to ${res.local_version || "custom"} (${res.binary_path || "auto"})`);
            notifyKernelBadgeListeners({
                ...info,
                installed: Boolean(res.binary_path),
                binary_path: res.binary_path,
                binary_source: res.binary_source,
                custom_binary_path: res.binary_source === "custom_path" ? res.binary_path : null,
                local_version: res.local_version,
            });
        } catch (err) {
            setStatus("error", `Failed to switch kernel: ${err.message}`);
        } finally {
            applyCustomBtn.disabled = false;
        }
    };

    const resetCustomBtn = document.createElement("button");
    resetCustomBtn.className = "pa-btn";
    resetCustomBtn.textContent = "Reset Auto";
    resetCustomBtn.onclick = () => {
        customInput.value = "";
        applyCustomBtn.click();
    };

    customRow.appendChild(customInput);
    customRow.appendChild(applyCustomBtn);
    customRow.appendChild(resetCustomBtn);
    secCustom.appendChild(customRow);
    manualPanel.appendChild(secCustom);

    if (info.bin_dir) {
        const secBin = document.createElement("div");
        secBin.className = "pa-manual-section";
        secBin.innerHTML = `<div class="pa-manual-label">3. Or Manually Unzip Into bin/ Folder</div>`;
        const binRow = document.createElement("div");
        binRow.className = "pa-manual-row";
        const binText = document.createElement("span");
        binText.className = "pa-asset-info";
        binText.style.flex = "1";
        binText.textContent = info.bin_dir;
        const copyBinBtn = document.createElement("button");
        copyBinBtn.className = "pa-btn";
        copyBinBtn.textContent = "Copy bin/ Path";
        copyBinBtn.onclick = async () => {
            try {
                await navigator.clipboard.writeText(info.bin_dir);
                copyBinBtn.textContent = "Copied ✓";
                setTimeout(() => (copyBinBtn.textContent = "Copy bin/ Path"), 1400);
            } catch (_) {}
        };
        const rescanBtn = document.createElement("button");
        rescanBtn.className = "pa-btn";
        rescanBtn.textContent = "Rescan";
        rescanBtn.onclick = () => checkKernelUpdate(false, true);
        binRow.appendChild(binText);
        binRow.appendChild(copyBinBtn);
        binRow.appendChild(rescanBtn);
        secBin.appendChild(binRow);
        manualPanel.appendChild(secBin);
    }

    body.appendChild(manualPanel);

    const footer = document.createElement("div");
    footer.className = "pa-modal-footer";

    const leftGroup = document.createElement("div");
    leftGroup.style.display = "flex";
    leftGroup.style.gap = "6px";

    const skipBtn = document.createElement("button");
    skipBtn.className = "pa-btn-secondary";
    skipBtn.textContent = "Skip this version";
    skipBtn.style.display = hasUpdate && info.latest_version ? "inline-block" : "none";
    skipBtn.onclick = async () => {
        skipBtn.disabled = true;
        try {
            await longFetch("/prompt_agent/skip_version", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ version: info.latest_version }),
            });
            notifyKernelBadgeListeners({ ...info, skipped: true, skipped_version: info.latest_version });
            closeUpdateModal();
        } catch (err) {
            skipBtn.disabled = false;
            setStatus("error", `Failed to skip version: ${err.message}`);
        }
    };

    const refreshBtn = document.createElement("button");
    refreshBtn.className = "pa-btn-secondary";
    refreshBtn.textContent = "Refresh Release";
    refreshBtn.onclick = async () => {
        refreshBtn.disabled = true;
        setStatus("info", "Fetching latest ggml-org/llama.cpp release metadata...");
        await checkKernelUpdate(true, true);
    };

    const cancelDlBtn = document.createElement("button");
    cancelDlBtn.className = "pa-btn-secondary";
    cancelDlBtn.textContent = "Cancel Download";
    cancelDlBtn.style.display = "none";
    cancelDlBtn.onclick = async () => {
        cancelDlBtn.disabled = true;
        try { await longFetch("/prompt_agent/cancel_update", { method: "POST" }); } catch (_) {}
    };

    leftGroup.appendChild(skipBtn);
    leftGroup.appendChild(refreshBtn);
    leftGroup.appendChild(cancelDlBtn);

    const rightGroup = document.createElement("div");
    rightGroup.className = "pa-modal-footer-right";

    const laterBtn = document.createElement("button");
    laterBtn.className = "pa-btn-secondary";
    laterBtn.textContent = "Close";
    laterBtn.onclick = closeUpdateModal;

    const updateBtn = document.createElement("button");
    updateBtn.className = "pa-btn-primary";
    updateBtn.textContent = !isInstalled ? "Install Kernel" : hasUpdate ? "Update Kernel" : "Reinstall / Switch Backend";
    updateBtn.disabled = backends.length === 0;

    const setDownloadingUI = (active) => {
        updateBtn.disabled = active || backends.length === 0;
        skipBtn.disabled = active;
        refreshBtn.disabled = active;
        select.disabled = active;
        cancelDlBtn.style.display = active ? "inline-block" : "none";
        cancelDlBtn.disabled = false;
        progressWrap.classList.toggle("visible", active);
    };

    const applyProgressSnapshot = (job) => {
        if (!job) return false;
        const st = job.state;
        if (st === "starting" || st === "downloading" || st === "extracting") {
            setDownloadingUI(true);
            const pct = Math.max(0, Math.min(100, Number(job.percent || 0)));
            progressFill.style.width = `${pct}%`;
            const dlStr = formatBytes(job.downloaded_bytes);
            const totStr = formatBytes(job.total_bytes);
            const spdStr = job.speed_bps > 0 ? `${(job.speed_bps / (1024 * 1024)).toFixed(1)} MB/s` : "";
            progressLeft.textContent = job.file_count > 0
                ? `File ${job.file_index}/${job.file_count}: ${job.current_file || ""}`
                : (job.stage || "Working...");
            progressRight.textContent = [
                dlStr && totStr ? `${dlStr} / ${totStr}` : dlStr,
                pct > 0 ? `${pct.toFixed(1)}%` : "",
                spdStr,
            ].filter(Boolean).join(" · ");
            setStatus("info", job.stage || `Downloading llama.cpp ${latestDisplay}...`);
            return true;
        }
        if (st === "done" && job.result) {
            stopUpdatePolling();
            setDownloadingUI(false);
            progressWrap.classList.add("visible");
            progressFill.style.width = "100%";
            progressLeft.textContent = "Complete";
            progressRight.textContent = "100%";
            const res = job.result;
            setStatus("success", `Installed llama.cpp ${res.installed_version} (${res.backend}) into bin/ successfully!`);
            notifyKernelBadgeListeners({
                ...info,
                installed: true,
                binary_path: res.binary_path,
                binary_source: "local_bin",
                local_version: res.installed_version,
                update_available: false,
                skipped: false,
            });
            updateBtn.textContent = "Installed ✓";
            return false;
        }
        if (st === "error") {
            stopUpdatePolling();
            setDownloadingUI(false);
            setStatus("error", `Update failed: ${job.error || "Unknown error"}`);
            return false;
        }
        if (st === "cancelled") {
            stopUpdatePolling();
            setDownloadingUI(false);
            setStatus("info", "Download cancelled.");
            return false;
        }
        return false;
    };

    const startProgressPolling = () => {
        stopUpdatePolling();
        const pollOnce = async () => {
            if (!document.getElementById(MODAL_ID)) {
                stopUpdatePolling();
                return;
            }
            try {
                const resp = await longFetch("/prompt_agent/update_progress");
                applyProgressSnapshot(await resp.json());
            } catch (_) {}
        };
        pollOnce();
        activeUpdatePollTimer = setInterval(pollOnce, 500);
    };

    updateBtn.onclick = async () => {
        setDownloadingUI(true);
        setStatus("info", `Starting background download for llama.cpp ${latestDisplay} (${selectedBackend})...`);
        try {
            const resp = await longFetch("/prompt_agent/update_kernel", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ backend: selectedBackend }),
            });
            const res = await resp.json();
            if (!resp.ok || !res.ok) throw new Error(res.error || `HTTP ${resp.status}`);
            if (res.job) applyProgressSnapshot(res.job);
            startProgressPolling();
        } catch (err) {
            setDownloadingUI(false);
            setStatus("error", `Update failed: ${err.message}`);
        }
    };

    longFetch("/prompt_agent/update_progress")
        .then((r) => r.json())
        .then((job) => {
            if (job && ["starting", "downloading", "extracting"].includes(job.state)) {
                applyProgressSnapshot(job);
                startProgressPolling();
            }
        })
        .catch(() => {});

    rightGroup.appendChild(laterBtn);
    rightGroup.appendChild(updateBtn);
    footer.appendChild(leftGroup);
    footer.appendChild(rightGroup);

    modal.appendChild(header);
    modal.appendChild(body);
    modal.appendChild(footer);
    backdrop.appendChild(modal);

    backdrop.addEventListener("click", (e) => {
        if (e.target === backdrop) closeUpdateModal();
    });
    document.body.appendChild(backdrop);
}

async function checkKernelUpdate(force = false, openIfUpToDate = false) {
    try {
        const url = force ? "/prompt_agent/check_update?force=1" : "/prompt_agent/check_update";
        const resp = await longFetch(url);
        const data = await resp.json();
        if (!data.ok) {
            if (openIfUpToDate) {
                showKernelUpdateModal({
                    installed: data.installed,
                    binary_path: data.binary_path,
                    binary_source: data.binary_source,
                    custom_binary_path: data.custom_binary_path,
                    bin_dir: data.bin_dir,
                    hardware: data.hardware,
                    official_releases_url: data.official_releases_url,
                    binary_release_url: data.binary_release_url,
                    local_version: data.local_version || "unknown",
                    latest_version: "unavailable",
                    update_available: false,
                    release_notes: `Version check failed: ${data.error || "Unknown network error"}`,
                    available_backends: [],
                });
            }
            return data;
        }
        notifyKernelBadgeListeners(data);
        if ((data.update_available && !data.skipped) || openIfUpToDate) {
            showKernelUpdateModal(data);
        }
        return data;
    } catch (err) {
        console.warn("[PromptAgent] Kernel update check failed:", err);
        return null;
    }
}

// ======================================================================
// Plan A: Minimalist Toolbar + Compact Post-Execution Result Echo Widget
// ======================================================================

function getWidgetValue(node, name, fallback) {
    const w = node.widgets?.find((item) => item.name === name);
    return w !== undefined ? w.value : fallback;
}

function updateComboOptions(node, widgetName, newValues) {
    if (!Array.isArray(newValues) || newValues.length === 0) return;
    const w = node.widgets?.find((item) => item.name === widgetName);
    if (!w || !w.options) return;
    w.options.values = newValues;
    if (!newValues.includes(w.value)) {
        w.value = newValues[0];
    }
}

function createPipelineFooterWidget(node) {
    injectStyles();

    const root = document.createElement("div");
    root.className = "pa-panel";
    stopCanvasPropagation(root);

    let lastItems = [];
    let activeIdx = 0;
    let isBusy = false;

    // --- 1-Line Compact Toolbar ---
    const toolbar = document.createElement("div");
    toolbar.className = "pa-toolbar";

    const statusWrap = document.createElement("div");
    statusWrap.className = "pa-status";
    const statusDot = document.createElement("span");
    statusDot.className = "pa-dot";
    const statusText = document.createElement("span");
    statusText.textContent = "Idle";
    statusWrap.appendChild(statusDot);
    statusWrap.appendChild(statusText);

    const btns = document.createElement("div");
    btns.className = "pa-toolbar-btns";

    const previewBtn = document.createElement("button");
    previewBtn.className = "pa-btn primary";
    previewBtn.textContent = "▶ Preview";
    previewBtn.title = "Test current transform_goal on connected image_1..6 without queueing downstream diffusion";

    const refreshBtn = document.createElement("button");
    refreshBtn.className = "pa-btn";
    refreshBtn.textContent = "↻";
    refreshBtn.title = "Rescan GGUF models, mmproj, prompt_skills/, system prompts, and WD14 models";

    const kernelBtn = document.createElement("button");
    kernelBtn.className = "pa-btn";
    kernelBtn.textContent = "⚙ Kernel";
    kernelBtn.title = "Check llama.cpp kernel version, auto-download, or switch binary";
    kernelBtn.onclick = () => checkKernelUpdate(true, true);

    const unloadBtn = document.createElement("button");
    unloadBtn.className = "pa-btn danger";
    unloadBtn.textContent = "🧹 Unload";
    unloadBtn.title = "Terminate llama-server immediately to free GPU VRAM";

    btns.appendChild(previewBtn);
    btns.appendChild(refreshBtn);
    btns.appendChild(kernelBtn);
    btns.appendChild(unloadBtn);
    toolbar.appendChild(statusWrap);
    toolbar.appendChild(btns);

    const updateKernelBadge = (info) => {
        if (!info) return;
        kernelBtn.classList.toggle("has-update", Boolean(info.update_available && !info.skipped));
        kernelBtn.title = info.installed
            ? `llama.cpp ${info.local_version || "?"} (latest: ${info.latest_display || info.latest_version || "?"})`
            : "llama-server not installed — click to install";
    };
    kernelBadgeListeners.add(updateKernelBadge);
    if (latestUpdateInfo) updateKernelBadge(latestUpdateInfo);

    const setStatus = (state, text) => {
        statusDot.className = `pa-dot ${state}`;
        statusText.textContent = text;
        statusWrap.title = text;
    };

    // --- Compact Result Echo Area ---
    const echoEl = document.createElement("div");
    echoEl.className = "pa-echo";

    const renderPlaceholder = () => {
        echoEl.innerHTML = `<div class="pa-echo-placeholder">Set <b>transform_goal</b> &amp; connect <b>image_1..6</b>, then click <b>Queue Prompt</b> (or <b>▶ Preview</b> to dry-run this node only).</div>`;
    };
    renderPlaceholder();

    const renderError = (msg) => {
        echoEl.innerHTML = "";
        const err = document.createElement("div");
        err.className = "pa-err-box";
        err.textContent = `Error: ${msg}`;
        echoEl.appendChild(err);
    };

    const renderResults = (items) => {
        if (!Array.isArray(items) || items.length === 0) {
            renderPlaceholder();
            return;
        }
        lastItems = items;
        if (activeIdx >= lastItems.length) activeIdx = 0;
        const cur = lastItems[activeIdx];

        echoEl.innerHTML = "";

        // Top meta row: batch pager + perf + Copy Prompt button
        const topRow = document.createElement("div");
        topRow.className = "pa-echo-top";

        const leftMeta = document.createElement("div");
        leftMeta.className = "pa-pager";
        if (lastItems.length > 1) {
            const prevBtn = document.createElement("button");
            prevBtn.className = "pa-btn";
            prevBtn.textContent = "‹";
            prevBtn.disabled = activeIdx === 0;
            prevBtn.onclick = () => { activeIdx = Math.max(0, activeIdx - 1); renderResults(lastItems); };

            const idxSpan = document.createElement("span");
            idxSpan.textContent = `${activeIdx + 1}/${lastItems.length}`;

            const nextBtn = document.createElement("button");
            nextBtn.className = "pa-btn";
            nextBtn.textContent = "›";
            nextBtn.disabled = activeIdx >= lastItems.length - 1;
            nextBtn.onclick = () => { activeIdx = Math.min(lastItems.length - 1, activeIdx + 1); renderResults(lastItems); };

            leftMeta.appendChild(prevBtn);
            leftMeta.appendChild(idxSpan);
            leftMeta.appendChild(nextBtn);
        }
        if (cur.perf) {
            const perfSpan = document.createElement("span");
            perfSpan.className = "pa-perf";
            perfSpan.textContent = cur.perf;
            leftMeta.appendChild(perfSpan);
        }

        const copyBtn = document.createElement("button");
        copyBtn.className = "pa-btn";
        copyBtn.textContent = "Copy Prompt";
        copyBtn.onclick = async () => {
            try {
                await navigator.clipboard.writeText(cur.final_prompt || "");
                copyBtn.textContent = "Copied ✓";
                setTimeout(() => (copyBtn.textContent = "Copy Prompt"), 1400);
            } catch (_) {}
        };

        topRow.appendChild(leftMeta);
        topRow.appendChild(copyBtn);
        echoEl.appendChild(topRow);

        if (cur.summary) {
            const sumEl = document.createElement("div");
            sumEl.className = "pa-summary-line";
            sumEl.textContent = `✦ ${cur.summary}`;
            echoEl.appendChild(sumEl);
        }

        if (Array.isArray(cur.stage_notes) && cur.stage_notes.length > 0) {
            for (const note of cur.stage_notes) {
                const nEl = document.createElement("div");
                nEl.className = "pa-note-line";
                nEl.textContent = `⚠ ${note}`;
                echoEl.appendChild(nEl);
            }
        }

        const promptBox = document.createElement("div");
        promptBox.className = "pa-prompt-box";
        promptBox.textContent = cur.final_prompt || "(empty prompt)";
        echoEl.appendChild(promptBox);

        // Collapsible 7 Visual Slots
        if (cur.slots && typeof cur.slots === "object" && Object.values(cur.slots).some((v) => String(v || "").trim())) {
            const slotDetails = document.createElement("details");
            slotDetails.className = "pa-details";
            const slotSum = document.createElement("summary");
            slotSum.textContent = "7 Visual Slots";
            slotDetails.appendChild(slotSum);

            const slotBody = document.createElement("div");
            slotBody.className = "pa-details-body";
            for (const def of SLOT_DEFS) {
                const val = String(cur.slots[def.key] || "").trim();
                if (!val) continue;
                const line = document.createElement("div");
                line.className = "pa-slot-line";
                const keySpan = document.createElement("span");
                keySpan.className = "pa-slot-key";
                keySpan.textContent = `${def.label}:`;
                line.appendChild(keySpan);
                line.appendChild(document.createTextNode(val));
                slotBody.appendChild(line);
            }
            slotDetails.appendChild(slotBody);
            echoEl.appendChild(slotDetails);
        }

        // Collapsible Tiered WD14 Tags
        if ((Array.isArray(cur.wd14_items) && cur.wd14_items.length > 0) || cur.wd14_tags) {
            const wdDetails = document.createElement("details");
            wdDetails.className = "pa-details";
            const wdSum = document.createElement("summary");
            wdSum.textContent = `WD14 Tiered Anchors${Array.isArray(cur.wd14_items) ? ` (${cur.wd14_items.length} img)` : ""}`;
            wdDetails.appendChild(wdSum);

            const wdBody = document.createElement("div");
            wdBody.className = "pa-details-body";
            if (Array.isArray(cur.wd14_items) && cur.wd14_items.length > 0) {
                for (const item of cur.wd14_items) {
                    const card = document.createElement("div");
                    card.className = "pa-slot-line";
                    const title = document.createElement("div");
                    title.className = "pa-slot-key";
                    title.textContent = `Image #${item.index}${item.name ? ` (${item.name})` : ""}`;
                    card.appendChild(title);
                    for (const t of [
                        { cls: "char", label: "Character", list: item.character },
                        { cls: "high", label: "Core ≥0.70", list: item.high },
                        { cls: "med", label: "Details", list: item.medium },
                    ]) {
                        if (!Array.isArray(t.list) || t.list.length === 0) continue;
                        const r = document.createElement("div");
                        const b = document.createElement("span");
                        b.className = `pa-tier-badge ${t.cls}`;
                        b.textContent = t.label;
                        r.appendChild(b);
                        r.appendChild(document.createTextNode(t.list.join(", ")));
                        card.appendChild(r);
                    }
                    wdBody.appendChild(card);
                }
            } else {
                wdBody.textContent = cur.wd14_tags;
            }
            wdDetails.appendChild(wdBody);
            echoEl.appendChild(wdDetails);
        }

        // Collapsible Reasoning
        if (cur.reasoning) {
            const rDetails = document.createElement("details");
            rDetails.className = "pa-details";
            const rSum = document.createElement("summary");
            rSum.textContent = "Reasoning";
            rDetails.appendChild(rSum);
            const rBody = document.createElement("div");
            rBody.className = "pa-details-body";
            rBody.textContent = cur.reasoning;
            rDetails.appendChild(rBody);
            echoEl.appendChild(rDetails);
        }
    };

    const refreshServerStatus = async () => {
        try {
            const resp = await longFetch("/prompt_agent/status");
            const data = await resp.json();
            if (data.running) {
                setStatus("running", data.model || "Running");
            } else {
                setStatus("", data.binary_installed ? "Idle" : "No Kernel");
            }
        } catch (_) {}
    };

    const setBusy = (busy, label) => {
        isBusy = busy;
        previewBtn.disabled = busy;
        refreshBtn.disabled = busy;
        unloadBtn.disabled = busy;
        if (busy) setStatus("busy", label || "Running...");
    };

    refreshBtn.onclick = async () => {
        if (isBusy) return;
        refreshBtn.disabled = true;
        try {
            const resp = await longFetch("/prompt_agent/models");
            const data = await resp.json();
            if (data.ok) {
                updateComboOptions(node, "llm_model", data.llm_models);
                updateComboOptions(node, "mmproj_model", data.mmproj_models);
                updateComboOptions(node, "output_style", data.output_styles);
                updateComboOptions(node, "system_prompt", data.system_prompts);
                updateComboOptions(node, "wd14_model", data.wd14_models);
                app.graph?.setDirtyCanvas(true, true);
            }
        } catch (err) {
            renderError(`Model refresh failed: ${err.message}`);
        } finally {
            refreshBtn.disabled = false;
        }
    };

    unloadBtn.onclick = async () => {
        if (isBusy) return;
        unloadBtn.disabled = true;
        try {
            await longFetch("/prompt_agent/unload", { method: "POST" });
            setStatus("", "Idle (VRAM freed)");
        } catch (err) {
            renderError(`Unload failed: ${err.message}`);
        } finally {
            unloadBtn.disabled = false;
        }
    };

    previewBtn.onclick = async () => {
        if (isBusy) return;
        setBusy(true, "Pulling connected images...");
        try {
            const images = await pullConnectedSocketImages(node);
            setBusy(
                true,
                images.length > 0
                    ? `WD14 (${images.length} img) → VLM → Transform...`
                    : "Running Prompt Agent..."
            );
            const autoUnload = Boolean(getWidgetValue(node, "auto_unload_vram", true));
            const payload = {
                images,
                llm_model: getWidgetValue(node, "llm_model", ""),
                mmproj_model: getWidgetValue(node, "mmproj_model", "none"),
                output_style: getWidgetValue(node, "output_style", "auto"),
                preserve_slots: getWidgetValue(node, "preserve_slots", "none"),
                transform_goal: getWidgetValue(node, "transform_goal", ""),
                batch_mode: getWidgetValue(node, "batch_mode", "auto"),
                wd14_model: getWidgetValue(node, "wd14_model", "wd-eva02-large-tagger-v3"),
                system_prompt: getWidgetValue(node, "system_prompt", "none"),
                auto_unload_vram: autoUnload,
                enable_wd14: getWidgetValue(node, "enable_wd14", true),
                wd14_threshold: getWidgetValue(node, "wd14_threshold", 0.35),
                wd14_character_threshold: getWidgetValue(node, "wd14_character_threshold", 0.85),
                exclude_tags: getWidgetValue(node, "exclude_tags", ""),
                max_tokens: getWidgetValue(node, "max_tokens", 2048),
                temperature: getWidgetValue(node, "temperature", 0.7),
                top_p: getWidgetValue(node, "top_p", 0.8),
                top_k: getWidgetValue(node, "top_k", 20),
                repeat_penalty: getWidgetValue(node, "repeat_penalty", 1.0),
                ctx_size: getWidgetValue(node, "ctx_size", 8192),
                seed: getWidgetValue(node, "seed", 1),
                timeout_seconds: getWidgetValue(node, "timeout_seconds", 300),
                reasoning: getWidgetValue(node, "reasoning", "off"),
                memory_mode: getWidgetValue(node, "memory_mode", "auto"),
                n_gpu_layers: getWidgetValue(node, "n_gpu_layers", 99),
                n_cpu_moe_layers: getWidgetValue(node, "n_cpu_moe_layers", 1),
                extra_args: getWidgetValue(node, "extra_args", ""),
            };

            const resp = await longFetch("/prompt_agent/preview", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });
            const data = await resp.json();
            if (!resp.ok || !data.ok) {
                throw new Error(data.error || `HTTP ${resp.status}`);
            }
            activeIdx = 0;
            renderResults(data.items || []);
            setStatus(autoUnload ? "" : "running", autoUnload ? "Preview Done (VRAM freed)" : (data.model_loaded || "Preview Ready"));
        } catch (err) {
            setStatus("error", "Preview Failed");
            renderError(err.message);
        } finally {
            setBusy(false);
        }
    };

    root.appendChild(toolbar);
    root.appendChild(echoEl);
    refreshServerStatus();

    const origOnRemoved = node.onRemoved;
    node.onRemoved = function () {
        kernelBadgeListeners.delete(updateKernelBadge);
        if (origOnRemoved) origOnRemoved.apply(this, arguments);
    };

    return {
        element: root,
        updateFromExecution(previewItems) {
            if (Array.isArray(previewItems) && previewItems.length > 0) {
                activeIdx = 0;
                renderResults(previewItems);
                const lastPerf = previewItems[previewItems.length - 1]?.perf;
                setStatus("", lastPerf ? `Done ${lastPerf}` : `Done (${previewItems.length} item${previewItems.length > 1 ? "s" : ""})`);
            }
        },
    };
}

// ======================================================================
// Universal Text & Reasoning Preview Widget (`PromptAgentTextPreview`)
// ======================================================================

function tryFormatText(raw) {
    const str = String(raw || "").trim();
    if ((str.startsWith("{") && str.endsWith("}")) || (str.startsWith("[") && str.endsWith("]"))) {
        try {
            return JSON.stringify(JSON.parse(str), null, 2);
        } catch (_) {}
    }
    return str;
}

function createUniversalTextPreviewWidget() {
    injectStyles();

    const root = document.createElement("div");
    root.className = "pa-panel";
    stopCanvasPropagation(root);

    let items = [];
    let activeIdx = 0;
    let viewMode = "split"; // "split" | "text" | "thinking"

    const toolbar = document.createElement("div");
    toolbar.className = "pa-toolbar";

    const leftPager = document.createElement("div");
    leftPager.className = "pa-pager";

    const modeBtns = document.createElement("div");
    modeBtns.className = "pa-toolbar-btns";

    const splitTab = document.createElement("button");
    splitTab.className = "pa-btn primary";
    splitTab.textContent = "Split";

    const textTab = document.createElement("button");
    textTab.className = "pa-btn";
    textTab.textContent = "Text";

    const thinkTab = document.createElement("button");
    thinkTab.className = "pa-btn";
    thinkTab.textContent = "Thinking";

    const copyTextBtn = document.createElement("button");
    copyTextBtn.className = "pa-btn";
    copyTextBtn.textContent = "Copy Text";

    const copyThinkBtn = document.createElement("button");
    copyThinkBtn.className = "pa-btn";
    copyThinkBtn.textContent = "Copy Think";

    modeBtns.appendChild(splitTab);
    modeBtns.appendChild(textTab);
    modeBtns.appendChild(thinkTab);
    modeBtns.appendChild(copyTextBtn);
    modeBtns.appendChild(copyThinkBtn);

    toolbar.appendChild(leftPager);
    toolbar.appendChild(modeBtns);

    const bodyEl = document.createElement("div");
    bodyEl.className = "pa-echo";

    const setTabMode = (mode) => {
        viewMode = mode;
        splitTab.classList.toggle("primary", mode === "split");
        textTab.classList.toggle("primary", mode === "text");
        thinkTab.classList.toggle("primary", mode === "thinking");
        render();
    };
    splitTab.onclick = () => setTabMode("split");
    textTab.onclick = () => setTabMode("text");
    thinkTab.onclick = () => setTabMode("thinking");

    const render = () => {
        leftPager.innerHTML = "";
        bodyEl.innerHTML = "";

        if (!Array.isArray(items) || items.length === 0) {
            const ph = document.createElement("div");
            ph.className = "pa-echo-placeholder";
            ph.textContent = "Connect any STRING output (and optional reasoning_in / meta_in) and run Queue Prompt to preview clean text & thinking.";
            bodyEl.appendChild(ph);
            copyThinkBtn.style.display = "none";
            return;
        }

        if (activeIdx >= items.length) activeIdx = 0;
        const cur = items[activeIdx];
        const cleanStr = tryFormatText(cur.clean_text);
        const thinkStr = String(cur.thinking || "").trim();
        const metaStr = String(cur.meta || "").trim();

        if (items.length > 1) {
            const prevBtn = document.createElement("button");
            prevBtn.className = "pa-btn";
            prevBtn.textContent = "‹";
            prevBtn.disabled = activeIdx === 0;
            prevBtn.onclick = () => { activeIdx = Math.max(0, activeIdx - 1); render(); };

            const idxSpan = document.createElement("span");
            idxSpan.textContent = `${activeIdx + 1}/${items.length}`;

            const nextBtn = document.createElement("button");
            nextBtn.className = "pa-btn";
            nextBtn.textContent = "›";
            nextBtn.disabled = activeIdx >= items.length - 1;
            nextBtn.onclick = () => { activeIdx = Math.min(items.length - 1, activeIdx + 1); render(); };

            leftPager.appendChild(prevBtn);
            leftPager.appendChild(idxSpan);
            leftPager.appendChild(nextBtn);
        }

        const charStat = document.createElement("span");
        charStat.className = "pa-perf";
        charStat.textContent = `${cleanStr.length} chars${thinkStr ? ` · Think ${thinkStr.length}` : ""}`;
        leftPager.appendChild(charStat);

        thinkTab.textContent = thinkStr ? "✦ Thinking" : "Thinking (0)";
        copyThinkBtn.style.display = thinkStr ? "inline-block" : "none";

        copyTextBtn.onclick = async () => {
            try {
                await navigator.clipboard.writeText(cleanStr);
                copyTextBtn.textContent = "Copied ✓";
                setTimeout(() => (copyTextBtn.textContent = "Copy Text"), 1400);
            } catch (_) {}
        };
        copyThinkBtn.onclick = async () => {
            try {
                await navigator.clipboard.writeText(thinkStr);
                copyThinkBtn.textContent = "Copied ✓";
                setTimeout(() => (copyThinkBtn.textContent = "Copy Think"), 1400);
            } catch (_) {}
        };

        if (metaStr) {
            const metaLine = document.createElement("div");
            metaLine.className = "pa-perf";
            metaLine.textContent = metaStr;
            bodyEl.appendChild(metaLine);
        }

        if ((viewMode === "split" || viewMode === "thinking") && thinkStr) {
            const thinkDetails = document.createElement("details");
            thinkDetails.className = "pa-details";
            thinkDetails.open = true;
            const sum = document.createElement("summary");
            sum.textContent = `Thinking / Reasoning (${thinkStr.length} chars)`;
            thinkDetails.appendChild(sum);

            const thinkBody = document.createElement("div");
            thinkBody.className = "pa-details-body";
            thinkBody.style.maxHeight = viewMode === "thinking" ? "260px" : "130px";
            thinkBody.style.whiteSpace = "pre-wrap";
            thinkBody.textContent = thinkStr;
            thinkDetails.appendChild(thinkBody);
            bodyEl.appendChild(thinkDetails);
        } else if (viewMode === "thinking" && !thinkStr) {
            const noThink = document.createElement("div");
            noThink.className = "pa-echo-placeholder";
            noThink.textContent = "No thinking / reasoning blocks detected in this output.";
            bodyEl.appendChild(noThink);
        }

        if (viewMode === "split" || viewMode === "text") {
            const textBox = document.createElement("div");
            textBox.className = "pa-prompt-box";
            textBox.style.maxHeight = viewMode === "text" ? "260px" : "150px";
            textBox.textContent = cleanStr || "(empty text)";
            bodyEl.appendChild(textBox);
        }
    };

    render();
    root.appendChild(toolbar);
    root.appendChild(bodyEl);

    return {
        element: root,
        updateItems(newItems) {
            items = Array.isArray(newItems) ? newItems : [];
            activeIdx = 0;
            render();
        },
    };
}

// ======================================================================
// ComfyUI Extension Registration
// ======================================================================

app.registerExtension({
    name: "ComfyUI.PromptAgent",

    async setup() {
        injectStyles();
        setTimeout(() => {
            checkKernelUpdate(false, false);
        }, 1500);
    },

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === "PromptAgent") {
            const origOnNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const result = origOnNodeCreated ? origOnNodeCreated.apply(this, arguments) : undefined;

                const ui = createPipelineFooterWidget(this);
                this._promptAgentUI = ui;

                const domWidget = this.addDOMWidget("prompt_agent_footer", "prompt_agent_ui", ui.element, {
                    serialize: false,
                    hideOnZoom: false,
                });
                domWidget.computeSize = (width) => [Math.max(width || 380, 380), 220];

                if (!this.size || this.size[0] < 400 || this.size[1] < 540) {
                    this.setSize([Math.max(this.size?.[0] || 420, 420), Math.max(this.size?.[1] || 580, 580)]);
                }

                return result;
            };

            const origOnExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                if (origOnExecuted) origOnExecuted.apply(this, arguments);
                if (message && Array.isArray(message.preview) && this._promptAgentUI) {
                    this._promptAgentUI.updateFromExecution(message.preview);
                }
            };
        } else if (nodeData.name === "PromptAgentTextPreview") {
            const origOnNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const result = origOnNodeCreated ? origOnNodeCreated.apply(this, arguments) : undefined;

                const ui = createUniversalTextPreviewWidget();
                this._textPreviewUI = ui;

                const domWidget = this.addDOMWidget("prompt_agent_text_preview", "prompt_agent_text_preview_ui", ui.element, {
                    serialize: false,
                    hideOnZoom: false,
                });
                domWidget.computeSize = (width) => [Math.max(width || 360, 360), 240];

                if (!this.size || this.size[0] < 380 || this.size[1] < 320) {
                    this.setSize([Math.max(this.size?.[0] || 400, 400), Math.max(this.size?.[1] || 350, 350)]);
                }

                return result;
            };

            const origOnExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                if (origOnExecuted) origOnExecuted.apply(this, arguments);
                if (message && Array.isArray(message.items) && this._textPreviewUI) {
                    this._textPreviewUI.updateItems(message.items);
                }
            };
        }
    },
});
