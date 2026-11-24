# ComfyUI-Prompt-Agent

[中文说明](#-中文说明) | [English](#-english)

---

## 📖 中文说明

**ComfyUI-Prompt-Agent** 是一个 **100% 独立自包含** 的 ComfyUI 视觉反推与智能提示词转换自定义节点包，底层由解耦的 **`llama-server` (`llama.cpp`)** 子进程与内置的 **`WD14 ONNX` 分级反推引擎** 驱动，**零第三方自定义节点依赖**。

它专为 ComfyUI 的 **「静态规则 · 动态数据流」** 范式设计：你只需在节点中写下一句固定的转换目标（例如：`把女生转换为 otokonoko`、`把图 1 的角色换上图 2 的服装与图 3 的姿势`，或者留空执行高保真原图反推），之后流经该节点的每一张图片或图片批次都会自动完成 **WD14 锚点提取 → VLM 视觉理解 → 7 维槽位消歧改写 → 自动释放显存**，并直接把改写好的无冲突英文提示词输出给下游采样器（Flux / Qwen-Image / Illustrious / SDXL / Flux Kontext / Qwen-Image-Edit 等）。

### ✨ 核心特性

- **零外部节点依赖，开箱即用**
  - 内置 **硬件自适应 `llama.cpp` 内核管理器（`⚙ Kernel`）**：自动识别显卡架构与驱动版本（NVIDIA Blackwell / Ada / Ampere CUDA 13.x / 12.x、Pascal、AMD ROCm / Vulkan、Intel、macOS Metal、CPU），支持一键后台异步下载官方 `llama-server` 内核（含 `cudart` 运行库），亦支持本地 `.zip` 离线安装。
  - 内置 **WD14 ONNX 分级反推引擎**：支持 `SmilingWolf` 主流 WD14 v3/v2 模型（默认 `wd-eva02-large-tagger-v3`），缺省自动从 HuggingFace 断点续传下载（支持 `HF_ENDPOINT` 镜像），并将标签按置信度分为 **Character（角色）**、**Core Anchors（核心特征 ≥0.70）** 与 **Details（细节）** 三级锚点。
- **两阶段显存零冲突调度（Two-Phase Batch Scheduler）**
  - **阶段 1（WD14 批量反推）**：单次加载 ONNX 会话处理完所有输入图片后，立即销毁会话并回收显存。
  - **阶段 2（VLM 视觉基线 + 7 维槽位转换）**：启动解耦的 `llama-server` 完成多模态理解与结构化 JSON 改写；当开启 `auto_unload_vram`（默认开启）时，节点执行完毕立即终止 `llama-server`，**将 100% 显存归还给下游扩散模型（KSampler）**，彻底杜绝爆显存（OOM）。
- **单指令驱动 + 冲突标签自动剔除**
  - 直接在 `transform_goal` 中用中文或英文描述转换意图（性转、换装、角色替换、跨图姿势迁移、画风迁移、自然语言扩写，或留空进行高精度原图反推）。
  - 遵循**最小侵入原则**：当属性发生变化时，自动删除原图中与之冲突的旧标签/旧描述，同时严格保留未要求修改的服装、姿势、背景与光影。
- **最多 6 图输入与智能批处理（`image_1` .. `image_6`）**
  - `batch_mode = auto`：单端口接入多帧 Batch 时自动逐张独立转换（`per_image`）；多端口同时接入参考图时自动进入跨图融合/迁移模式（`combine_multi`）。
- **多下游风格适配（`output_style`）**
  - `auto` / `natural_language`（适合 Flux / Qwen-Image / HiDream 自然语言长句）
  - `danbooru_tags`（适合 Illustrious / NoobAI / Pony / SDXL 纯标签流）
  - `hybrid`（核心标签锚点 + 自然语言描述）
  - `edit_instruction`（适合 Flux Kontext / Qwen-Image-Edit 的 `Transform... while keeping...` 编辑指令）
  - `structured_slots` / `custom_preset` 及 `prompt_skills/*.md|txt` 扩展技能
- **附带通用思考与文本预览节点 `Text & Reasoning Preview (Universal)`**
  - 可接入 ComfyUI 中任意节点的 `STRING` 输出，自动剥离并分栏展示 `<think>...</think>` / `<reasoning>...</reasoning>` 等思考过程，向下游输出干净的 `clean_text`。

---

### 📦 安装指南

1. 进入你的 `ComfyUI/custom_nodes/` 目录并克隆本仓库：
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/mizukir0418-gif/ComfyUI-Prompt-Agent.git
   ```
2. 安装 Python 依赖（ComfyUI 便携版请使用内置 `python_embeded`）：
   ```bash
   # 标准环境
   pip install -r ComfyUI-Prompt-Agent/requirements.txt

   # Windows 便携版 (ComfyUI_windows_portable)
   ..\..\..\python_embeded\python.exe -m pip install -r requirements.txt
   ```
3. **放置 GGUF 模型**：
   - 将你的 `.gguf` 语言/多模态模型与 `mmproj-*.gguf` 视觉投影权重放入 `ComfyUI/models/LLM/` 目录（自定义系统提示词可放入 `ComfyUI/models/LLM/prompts/*.txt`）。
4. **下载 `llama.cpp` 内核与 `WD14` 模型**：
   - 启动 ComfyUI，添加 **`Prompt Agent (llama.cpp + WD14)`** 节点。
   - 点击节点底部工具栏的 **`⚙ Kernel`** 按钮，一键下载适配你当前显卡的 `llama-server` 内核（或手动选择本地下载好的官方 Release `.zip` 包）。
   - 首次运行时，若本地没有所选的 WD14 模型，节点会自动下载至 `ComfyUI-Prompt-Agent/models/wd14/`（也可复用 `ComfyUI/models/wd14_tagger/` 中已有的 `.onnx` 与 `.csv` 文件）。

---

### 🧩 节点参数与端口说明

#### 1. `Prompt Agent (llama.cpp + WD14)`
| 参数 / 端口 | 类型 | 说明 |
| :--- | :--- | :--- |
| `llm_model` | Combo | 选择 `ComfyUI/models/LLM/` 下的主模型 `.gguf` |
| `mmproj_model` | Combo | 选择配套的视觉投影 `mmproj-*.gguf`（选 `none` 则仅使用 WD14 + 纯文本 LLM） |
| `output_style` | Combo | 输出提示词格式：`auto`、`natural_language`、`danbooru_tags`、`hybrid`、`edit_instruction`、`structured_slots`、`custom_preset` |
| `preserve_slots` | Combo | 硬性锁定的视觉槽位（默认 `none`，按指令自动保留未提及维度） |
| `transform_goal` | String | **核心转换目标**：填写自然语言改写规则；留空则默认执行高保真原图反推 |
| `batch_mode` | Combo | `auto`（单端口多图逐张处理 / 多端口跨图融合）、`per_image`、`combine_multi` |
| `wd14_model` | Combo | 内置 WD14 ONNX 模型选择（默认 `wd-eva02-large-tagger-v3`） |
| `system_prompt` | Combo | 可选加载 `ComfyUI/models/LLM/prompts/*.txt` 自定义系统提示词 |
| `image_1` .. `image_6` | `IMAGE` (可选) | 最多支持 6 路图像输入 |
| `goal_override` | `STRING` (可选) | 可由上游节点动态传入转换目标覆盖 `transform_goal` |
| **输出 `final_prompt`** | `STRING` | 最终英文提示词（可直接连入 `CLIPTextEncode`） |
| **输出 `structured_slots`** | `STRING` | 7 维视觉槽位 JSON（`subject`, `clothing`, `pose_expression`, `background`, `lighting`, `composition`, `style`） |
| **输出 `wd14_tags`** | `STRING` | 分级 WD14 客观锚点标签 |
| **输出 `reasoning`** | `STRING` | 模型的思考推理过程（开启 `reasoning` 时） |
| **输出 `perf`** | `STRING` | `llama-server` 预填充（Prompt t/s）与生成速度（Generation t/s）统计 |

#### 2. `Text & Reasoning Preview (Universal)`
- **输入**：`text`（任意文本输出）、`reasoning_in`（可选独立思考输入）、`meta_in`（可选速度或标签元信息）、`strip_thinking_from_output`（默认 `True`，自动剥离思考标签）。
- **输出**：`clean_text`（剥离思考后的纯净文本）、`thinking`（提取出的思考内容）、`raw_text`（原始文本）。

---

## 🌐 English

**ComfyUI-Prompt-Agent** is a **100% standalone, self-contained** ComfyUI custom node package for vision reverse-captioning and rule-driven prompt transformation. It pairs a decoupled **`llama-server` (`llama.cpp`)** subprocess with a built-in **`WD14 ONNX` tiered tagger**, requiring **zero external third-party custom nodes**.

### ✨ Key Highlights

- **Static Rules, Dynamic Dataflow**: Write your transformation goal once in `transform_goal` (e.g., gender/presentation swap, character replacement, outfit swap, pose transfer across `image_1..6`, style transfer, or leave empty for high-fidelity reverse captioning). Every image or batch flowing through the node is automatically analyzed and transformed into a conflict-free prompt.
- **Built-in GPU-Aware `llama.cpp` Kernel Downloader (`⚙ Kernel`)**: Automatically detects your GPU architecture and CUDA/ROCm/Vulkan/Metal driver to download official `ggml-org/llama.cpp` releases asynchronously in the background, with full offline `.zip` manual installation support.
- **Built-in Self-Contained WD14 ONNX Engine**: Auto-downloads SmilingWolf WD14 v3/v2 models with HTTP Range resume and partitions tags into **Character**, **Core Anchors (`>= 0.70`)**, and **Details** tiers.
- **Two-Phase Zero-VRAM-Overlap Scheduling**: Runs WD14 batch tagging first and frees ONNX VRAM before launching `llama-server`; when `auto_unload_vram` is enabled (default), `llama-server` terminates immediately after prompt generation so downstream diffusion models (`Flux`, `Qwen-Image`, `SDXL`, `Illustrious`) get 100% of your GPU VRAM.
- **Universal Text & Reasoning Preview Node**: Separates inline `<think>` / `<reasoning>` blocks from any `STRING` output in ComfyUI and forwards clean prompts downstream.

## 📄 License

Released under the [MIT License](LICENSE).
