# ComfyUI-NInfer

简体中文 | **[English](README.md)**

把 [NInfer](https://github.com/Neroued/ninfer) 大模型跑进 ComfyUI **自己的进程**。两个硬优势：

- **比同尺寸模型快 4~7 倍** —— 与本地 llama.cpp 跑 27B 级模型的逐项实测：扩写提示词 **5.6×**、
  翻译 **4.0×**、4 镜头分镜 **7.0×**（原始数据见下文实测对比），decode 约 70 tok/s。
- **不需要装任何 LLM 服务** —— 不装 Ollama / llama-server，不起 HTTP 服务，不开子进程，零 pip 依赖。
  引擎是原生 DLL，通过 ctypes 直接加载进 ComfyUI：只加载一次并常驻，之后每次调用开销 0 秒。
- **视觉：图片 + 整段视频** —— 最多 8 个图片口，接一个亮一个；另有 VIDEO 口，由引擎自带 FFmpeg
  解码（按模型配置的 fps 抽帧 + 时间位置编码，模型能描述视频内容和先后顺序）。名字带 "i2i" 的
  模型自动套提示词工程师人设；其他模型不带任何系统提示词，除非你手动填。

![两个节点](docs/images/nodes.png)

## 环境要求

| | |
|---|---|
| 系统 | Windows x64 |
| 显卡 | NVIDIA **RTX 50 系**（`sm_120`）或 **RTX 40 系**（`sm_89`）——**多图识别与视频理解只有这两类构建支持**；30/20 系社区构建是旧版单图引擎（见下表） |
| 显存 | 约 15 GiB 的制品**最低 16 GB**；官方制品需要 24 GB 以上 |
| ComfyUI | 近期版本均可（开发环境 0.36） |
| Python 包 | **无** —— numpy 和 Pillow 随 ComfyUI 自带 |
| 磁盘 | 引擎约 250 MB + 模型制品 16~23 GB |

引擎二进制必须和显卡的 compute capability 匹配。现有四份预编译构建：

| 显卡 | Compute | 预编译引擎 | 其他显卡 |
|---|---|---|---|
| RTX 5090 / 5080 / 5070 Ti / 5070 | `sm_120` | ✅ [官方构建](https://github.com/YukinoKaorisuna/ComfyUI-NInfer/releases)（`sm_120a`） | — |
| RTX 40 系（4050 Laptop ~ 4090） | `sm_89` | ✅ [社区构建](https://github.com/YukinoKaorisuna/ComfyUI-NInfer/releases)（`sm89` 包），40 系全系通用 | L4 / L40S 同为 `sm_89`，理论可用、未验证 |
| RTX 30 系（3060 ~ 3090 Ti）、A40 | `sm_86` | ⚠️ [社区构建](https://github.com/YukinoKaorisuna/ComfyUI-NInfer/releases)（`sm86` 包）—— **未在实机验证**。**旧单图引擎：无多图、无视频** | — |
| RTX 20 系（2060 ~ 2080 Ti） | `sm_75` | ⚠️ [社区构建](https://github.com/YukinoKaorisuna/ComfyUI-NInfer/releases)（`sm75` 包）—— **未在实机验证**，源自[社区 Turing 移植](https://github.com/mr-september/ninfer-2080ti-22g)。**旧单图引擎：无多图、无视频** | GTX 10 系及更老不支持 |

> [!NOTE]
> `sm86` 包是在 Blackwell 机器上交叉编译的（构建时没有 30 系卡可用）：Ampere 没有 FP8 tensor core，
> 所以 FP8 A8 / NVFP4 / TMA 三族 kernel 被编译期排除，改走 A16 反量化与 W8/Q4-Q6 路径。
> 已确认 fatbin 内含原生 sm_86 cubin、引擎能完成模型加载与全部工作区分配；**未在 30 系实机跑过生成**。
| A100 / H100 | `sm_80` / `sm_90` | ❌ | [自行编译](docs/COMPATIBILITY.md#other-gpus) |
| AMD、Intel、Apple | — | ❌ | 不支持（引擎只支持 CUDA） |

在不支持的卡上会报 `CUDA error: no kernel image is available for execution on the device`，
节点会直接给出**针对你这张卡**的编译命令。
详细架构矩阵、验证状态、各架构的基础仓库见 [docs/COMPATIBILITY.md](docs/COMPATIBILITY.md)。

## 安装

```bash
cd <ComfyUI>/custom_nodes
git clone https://github.com/YukinoKaorisuna/ComfyUI-NInfer
cd ComfyUI-NInfer
python tools/fetch_engine.py
```

ComfyUI 便携版：

```bat
cd ComfyUI_windows_portable
git clone https://github.com/YukinoKaorisuna/ComfyUI-NInfer ComfyUI\custom_nodes\ComfyUI-NInfer
python_embeded\python.exe ComfyUI\custom_nodes\ComfyUI-NInfer\tools\fetch_engine.py
```

`fetch_engine.py` 会识别你的显卡，把对应的引擎包（约 230~300 MB）下载到 `bin/`——Releases 里
每个显卡系列一个独立的包（50 系 `sm120a`、40 系 `sm89`），脚本自动挑对的那一个；**检测到已安装
就自动跳过**（`--force` 强制重下）。之后**重启 ComfyUI**。想先确认环境：

```
python tools/doctor.py
```

也可以手动下载：到 [Releases](https://github.com/YukinoKaorisuna/ComfyUI-NInfer/releases) 页面，
取与显卡匹配的 `ninfer-engine-win-x64-<架构>-<版本>.zip`（例如 `sm120a`），把压缩包里的全部文件
解压到 `ComfyUI-NInfer/bin/`。

### RTX 40 系（sm_89）

40 系用的是社区构建：在 [Releases](https://github.com/YukinoKaorisuna/ComfyUI-NInfer/releases)
里下载 `ninfer-engine-win-x64-sm89-0.1.0.zip`（约 291 MB）——或者直接跑上面的
`python tools/fetch_engine.py`，它会自动选中这个包。手动安装时——

1. **把压缩包里的全部 21 个 DLL 解压到 `ComfyUI-NInfer/bin/`**，不是只放引擎那一个文件。包内是
   引擎 `ninfer_capi.dll` + 10 个运行时依赖（FFmpeg 7.x / libcurl 等）+ 10 个 VC++ 运行时，
   **缺任何一个引擎都无法加载**（报缺依赖 / DLL not found）。
2. 节点的 `dll_path` 控件**保持留空** —— 引擎自动从 `bin/` 解析。之前填过旧路径的记得清空：
   指向不存在的文件时会被静默跳过，指向旧文件时会加载旧引擎。
3. 重启 ComfyUI。启动日志横幅应显示 `[NInfer] Ready - NVIDIA GeForce RTX 4xxx … (sm_89)`。

> [!NOTE]
> 这个 zip 自带全部依赖：解压即用，不需要 CUDA Toolkit，也不需要再单独下载任何 DLL。
> 50 系用户请用同页的 `sm120a` 包，别混用 —— 两个引擎只各认自己的架构。

```
ComfyUI-NInfer/
├── bin/          引擎及其运行时 DLL 落在这里（已被 git 忽略）
└── tools/        fetch_engine.py、doctor.py、build_engine.ps1
```

## 模型

只认 `.ninfer`。**GGUF 和 safetensors 不支持** —— 那是另一套引擎。

> [!IMPORTANT]
> 有两个不同代的制品**同名，都叫 `qwen3_8_27b.ninfer`**，但**不是同一个文件**（15.33 GiB 与
> 19.03 GiB）。按自己的显存下对的那个；两个都留就分开放。

### 8 GB 卡（唯一可用档）

| 文件 | 体积 | 仓库 |
|---|---|---|
| `qwen3_8_9b.ninfer` | 6.07 GiB | [YukinoKaorisuna/Qwen3.8-9B-ninfer-8gb](https://huggingface.co/YukinoKaorisuna/Qwen3.8-9B-ninfer-8gb) |
| `qwen3_8_9b_uncensored.ninfer` | 6.07 GiB | [YukinoKaorisuna/Qwen3.8-9B-Uncensored-ninfer-8gb](https://huggingface.co/YukinoKaorisuna/Qwen3.8-9B-Uncensored-ninfer-8gb) |

**这是 8 GB 显卡目前唯一能用的 .ninfer 制品** —— 其余已发布制品都是 27B 级（15.33 / 19.03 GiB），要 16 GB 起，
8 GB 卡放不下。实测（RTX 5070 Ti，`max_context` 2048）：显存 **4.54 GiB**、decode **114 tok/s**，
放进 `ComfyUI/models/LLM/` 即可被节点扫描到。9B 显存余量够，不用像 27B 那样靠 `NInfer Free VRAM` 分时复用。

来源：Qwen3.8-9B 蒸馏版（[empero-ai/Qwen3.8-9B-Distill](https://huggingface.co/empero-ai/Qwen3.8-9B-Distill)，
沿用 Qwen3.5-9B 架构），破限版取自 [nurdich/…-uncensored-heretic](https://huggingface.co/nurdich/Qwen3.8-9B-Distill-uncensored-heretic)。

### 16 GB 卡（本文档针对这个）

| 文件 | 体积 | 仓库 |
|---|---|---|
| `qwen3_8_27b.ninfer` | **15.33 GiB** | [ninfer-5080/Qwen3.8-27B-RTX5080](https://huggingface.co/ninfer-5080/Qwen3.8-27B-RTX5080) |

本文档所有显存与速度数字都是用这个文件测的，预编译引擎也对着它验证。它是**专为 16 GB 卡发布的
配方**：Q3/Q4/Q5 混合权重（约 3.95 BPW），128K 上下文、Q4 KV、MTP-3、视觉全部验证过。

下载后核对：

```
sha256sum qwen3_8_27b.ninfer
# 必须等于
c4a7e9ab593a7f42d58208fa0065d67a82d61921107686cc9f6ed1ec6b050e21
```

### 24 GB 及以上

精度更高、体积更大的制品，由 NInfer 作者 [Neroued](https://huggingface.co/neroued) 发布
（Apache-2.0，均带视觉与 MTP）：

| 文件 | 体积 | 仓库 |
|---|---|---|
| `qwen3_6_27b.ninfer` | 16.29 GiB | [neroued/Qwen3.6-27B-NInfer](https://huggingface.co/neroued/Qwen3.6-27B-NInfer) |
| `qwen3_8_27b.ninfer` | 19.03 GiB | [neroued/Qwen3.8-27B-NInfer](https://huggingface.co/neroued/Qwen3.8-27B-NInfer) |
| `qwen3_6_35b_a3b.ninfer` | 21.23 GiB | [neroued/Qwen3.6-35B-A3B-NInfer](https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer) |
| `qwen3_8_27b_nvfp4.ninfer` | 22.09 GiB | ⚠️ NVFP4 在 Windows 上不可用 |
| `qwen3_6_27b_nvfp4.ninfer` | 17.07 GiB | ⚠️ 同上 |

### 社区制品

| 文件 | 仓库 | 说明 |
|---|---|---|
| `ornith_1_5_35b_a3b.ninfer` | [huggingJDE/Ornith-1.5-35B-A3B-NInfer](https://huggingface.co/huggingJDE/Ornith-1.5-35B-A3B-NInfer) | RTX 3090 / 3080 唯一可用 |
| `qwen3_8_27b_uncensored.ninfer` | [YukinoKaorisuna/Qwen3.8-27B-Uncensored-ninfer](https://huggingface.co/YukinoKaorisuna/Qwen3.8-27B-Uncensored-ninfer) | 破限版 |

把文件放进 **`ComfyUI/models/LLM/`**，重启 ComfyUI，它就会出现在节点的 `model` 下拉里。追加目录：

```bat
set NINFER_MODEL_DIRS=D:\models
```

完整清单、镜像、容器版本注意事项见 [docs/MODELS.md](docs/MODELS.md)。

## 节点

### NInfer Local LLM (.ninfer / Qwen3.8)

`image` / `images` / `image3`…`image8` 都是可选输入：连一张或多张图做反推——新接口会随连线逐个
出现（最多 8 个图片批口）。`video` 接 VIDEO（来自 Load Video）：引擎自带 FFmpeg 解码，按模型
配置的 fps 抽帧并加时间位置编码，模型能描述视频内容和先后顺序。所有口可自由组合，一次请求
发送，图片按顺序编号排在文本前。

| 参数 | 默认 | 说明 |
|---|---|---|
| `model` | 上次成功用的那个 | 自动扫描 `ComfyUI/models/LLM`（含插件自带 `models/`）里的 `.ninfer` —— 文件放进去、重启 ComfyUI 就出现，**不用手动输路径** |
| `system_prompt` | *（空 = 自动）* | 留空 = AUTO：仅名字含 "i2i" 的模型自动套提示词工程师人设，其他模型**不带**任何系统提示词；手动填的内容永远优先 |
| `user_prompt` | | 你的输入 |
| `max_context` | 4096 | 显存第一杠杆；16 GB 上 4096 是稳的上限 |
| `max_tokens` | 512 | |
| `temperature` `top_k` `top_p` `min_p` `presence_penalty` `frequency_penalty` `seed` | 0.7 / 20 / 0.8 / 0 / 1.5 / 0 / −1 | 采样参数 |
| `enable_thinking` | **false** | 保持关 —— 提示词任务上差 5~10 倍 |
| `vision` | true | 连图时必须开；多占约 2 GiB |
| `mtp_draft_tokens` | 0 | 设 3 开启 MTP 投机解码 |
| `kv_dtype` | bf16 | 显存紧就改 `int8` |
| `embedding_host` | true | embedding 表放主机内存，省约 0.8 GiB，保持开 |
| `image_max_side` | 1024 | 输入的图会缩到这个尺寸 |
| `keep_loaded` | true | 关掉 = 每次生成完释放引擎 |
| `use_cuda_graph` | true | 最快；取消勾选可解锁更大的 `max_context` |
| `free_comfy_vram` | true | 创建引擎前卸载 ComfyUI 自己的模型 |
| `auto_recover` | true | 创建失败时自动重试而不是直接报错（见常见报错） |
| `model_path_override`、`dll_path` | 空 | 仅当文件不在扫描目录里才用；**留空**则引擎自动从 `bin/` 解析 |

插件有记忆：一次成功生成后，新拖的节点默认选中该模型；`info` 输出会显示实际加载的引擎
路径（`engine=... dll_path widget was auto-resolved`）。已保存的工作流仍用自己存的值。

输出 `text` 和 `info`。`info` 会报耗时和引擎的显存占用，第一次跑建议接一个 Show Text 节点。

`max_context`、`vision`、`kv_dtype`、`embedding_host`、`mtp_draft_tokens`、`use_cuda_graph`
是**创建引擎时固定**的，改动任一项会触发一次重载（几秒）。

### NInfer Free VRAM (pass-through)

可以插在任意连线上 —— `anything` 接受任何类型，`output` 把同一个值原样透传，所以能夹在采样器和
任何东西之间。什么都不连时就是独立的清理节点。

| 参数 | 说明 |
|---|---|
| `offload_model` | 卸载 ComfyUI 的模型 |
| `offload_cache` | 清空 torch 缓存 |

它还会释放引擎的约 14 GiB。这件事别的节点做不到：那部分是 DLL 里的原生 `cudaMalloc`，
通用的「清理显存」节点看不见它。

## 推荐配置

| 目标 | 设置 |
|---|---|
| 扩写 / 翻译提示词 | `max_context 4096`、`embedding_host ✓`、`enable_thinking ✗`，从不连图就 `vision ✗` |
| 看图反推 | 同上，另外 `vision ✓` 并连上 `image` |
| 需要更大上下文 | `use_cuda_graph ✗`，再 `max_context 8192`（代价是 decode 速度约减半） |
| 生成完释放显存 | `keep_loaded ✗`，或下游插 `NInfer Free VRAM` |

## 显存

> [!IMPORTANT]
> **引擎占用的显存对 ComfyUI 自带的清理是不可见的——必须用专用的释放方式。** 引擎的显存是 DLL
> 内部原生 `cudaMalloc` 分配的，ComfyUI 的「清理显存」按钮和 torch 的缓存清空都碰不到它。真正
> 要释放显卡：运行 **NInfer Free VRAM 节点**（单独跑，或串在你采样器前面的连线上），或者把
> LLM 节点的 `keep_loaded` 关掉。忘了这一步是「ComfyUI 模型都卸了、显存还显示占着 ~14 GiB」
> 的最常见原因。

27B 制品在 16 GB 卡上是**贴着上限**的。ComfyUI 自身还占约 1.3 GiB，所以成败由几百 MiB 决定。
以下是在 RTX 5070 Ti（15.92 GiB）上、ComfyUI 同时运行、制品 15.33 GiB 的实测：

| `max_context` | `embedding_host` | `use_cuda_graph` | 结果 |
|---|---|---|---|
| 8192 | 开 | 开 | ❌ 失败 —— 差 0.09 GiB |
| 8192 | 开 | **关** | ✅ 能跑，但慢（62 chars/s） |
| 6144 | 开 | 开 | ⚠️ 时好时坏 —— graph 预算间歇性超标 |
| 6144 | 开 | 关 | ✅ 能跑 |
| **4096** | **开** | **开** | ✅ **推荐** —— 最快（128 chars/s） |

- 只有 16 GB 的话，**务必打开 `embedding_host`**（白省约 0.8 GiB），`max_context` 保持 4096。
- 报 `CUDA Graph preparation consumed …` 时，解法是把 `use_cuda_graph ✗`，而不是去腾更多显存。
- 这个尺寸下引擎和扩散模型无法共存。用 `NInfer Free VRAM` 分时复用，或换更小（约 11 GiB）的制品。

## 常见报错

| 报错 | 含义 | 怎么办 |
|---|---|---|
| `no kernel image is available for execution on the device` | 引擎里没有你这张卡的代码 | [按你的架构编译](docs/COMPATIBILITY.md#other-gpus) |
| `Qwen3.6 family runtime requires compute capability 12.0` | 加载到的还是旧引擎 —— 40 系新构建已放宽为 `sm_89+` | 用新 zip 里的 `ninfer_capi.dll` 覆盖 `bin/` 的那份，并清空 `dll_path`，重启 |
| `runtime reservation requires X, but only Y bytes are available` | `Y` 是权重加载后的实时空闲显存；`0` 就是一字节不剩 | `free_comfy_vram ✓`、降 `max_context`、`kv_dtype int8`、`vision ✗` |
| `model weights require X … but only Y bytes are free` | 权重本身就放不下 | 腾显存或换更小的制品 |
| `CUDA Graph preparation consumed X, exceeding the planned allowance of Y` | graph 预算对这个配置算小了 | `use_cuda_graph ✗`（开了 `auto_recover ✓` 会自动重试） |

**Engine not found** —— 跑 `python tools/fetch_engine.py`。
**Model not found** —— 报错里会列出所有扫描过的目录。
**刚放的模型下拉里没有** —— 重启 ComfyUI，列表是启动时构建的。
**生成很慢** —— 检查 `enable_thinking` 是不是关了。
**用完 LLM 后 ComfyUI 爆显存** —— 引擎还驻留着；在采样器前插 `NInfer Free VRAM`，或 `keep_loaded ✗`。
**图像是按 JPEG 送进去的** —— 打包的 FFmpeg 没编 PNG 解码器。

`python tools/doctor.py` 会把上面这些一次性全查一遍。

## 实测性能

RTX 5070 Ti / 16 GB / ComfyUI 同时运行 / `max_context 4096`：

| | |
|---|---|
| 引擎创建（15.33 GB 制品） | 3.3 秒（缓存热），冷启约 10 秒 |
| 引擎复用 | **0.00 秒** |
| Decode | 约 70 tok/s（48 token 实测 0.68 秒） |
| 跑完 `NInfer Free VRAM` | 显存完整归还 |

Python 那层包装每次执行约 8 µs —— 占一次生成不到 0.002%，原生路径未变，所以 decode 速度与单独
运行引擎完全一致。

### 与 llama.cpp 节点的实测对比

同一台机器、同为 27B 级别的模型、`max_context 4096`、两边输入逐字节一致，并且**每次运行都用随机
seed** —— 输入不变时 ComfyUI 会直接复用缓存输出，seed 固定的话测的就是缓存而不是引擎：

| 任务（输出字符数） | 本包 | `ComfyUI-LLM-text-processor`（llama.cpp，Qwen3.8-27B IQ4_XS 14.63 GiB） | 提速 |
|---|---|---|---|
| 扩写提示词（约 215） | **9.1 秒** | 51.5 秒 | **5.6×** |
| 翻译提示词（约 250） | **7.1 秒** | 28.3 秒 | **4.0×** |
| 4 镜头分镜（约 570） | **14.3 秒** | 99.7 秒 | **7.0×** |
| 8 镜头脚本、2048 token 预算（约 5100） | **58.9 秒** | 600 秒内没跑完 | — |

任务越大差距越大：短提示词快几倍，完整分镜脚本快一个数量级。冷启动（首次运行、读 15 GB）是
16.5 秒 对 63.8 秒。

两个引擎都约 15 GB，所以**同一条工作流里装两套 LLM 后端会互相挤掉对方** —— 切换使用时必然重新加载。

> [!WARNING]
> **不要把清理节点放在 LLM 节点前面。** 工作流里如果 `NInfer Free VRAM`（或任何显存清理节点）
> 在 LLM **之前**，会把常驻引擎销毁，每次都强制重新加载 15 GB —— 实测从 7~14 秒退化到
> **16~25 秒**。把它放在 LLM 节点**之后**，或者干脆不用、保持 `keep_loaded` 开启。

### 与常见替代方式对比

| | 本包（进程内 DLL） | 本地 HTTP 服务（Ollama / llama-server） | 每次冷加载的节点 |
|---|---|---|---|
| 调用开销 | 一次 ctypes 函数调用 | HTTP + JSON 往返 | 进程启动 |
| 首次就绪 | 3~10 秒 | 3~10 秒 | **每次都要 3~10 秒** |
| 之后的每次运行 | **0 秒**（引擎常驻） | 0 秒，但 ~15 GB 由第二个进程持有、要另外管理 | 0 秒 |
| 显存归还 | `NInfer Free VRAM` 把显存一字不差还给扩散模型 | 服务进程一直握着，要额外手段 | 进程退出时 |
| 额外依赖 | 无 | 多装并维护一个服务 | 通常要装 pip 包 |

## 更多文档

- [docs/COMPATIBILITY.md](docs/COMPATIBILITY.md) —— 显卡 / 系统矩阵、其他架构的编译方法
- [docs/MODELS.md](docs/MODELS.md) —— 完整制品清单、镜像、容器版本
- [docs/TECHNICAL.md](docs/TECHNICAL.md) —— 进程内桥接是怎么搭的

## 致谢

- **[Neroued](https://github.com/Neroued)** —— NInfer 引擎作者，官方制品的发布者。本包只是它的客户端。
- **[toddballinger](https://github.com/toddballinger/ninfer-5080)** —— RTX 5080 优化线，以及本文档
  所依赖的那个 15.33 GiB / 16 GB 制品的发布者。
- **[YukinoKaorisuna/ninfer-5070ti](https://github.com/YukinoKaorisuna/ninfer-5070ti)** ——
  预编译引擎所基于的 Windows/MSVC 移植（[上游 PR](https://github.com/toddballinger/ninfer-5080/pull/16)）。
- **其他架构的社区引擎构建**：[Ambolio](https://github.com/Ambolio/ninfer-4090-windows)、
  [UDPSendToFailed](https://github.com/UDPSendToFailed/ninfer-4090)、
  [Don-Chad](https://github.com/Don-Chad/ninfer-3090)、
  [natpate](https://github.com/natpate/ninfer-windows)。
- **[阿里通义千问](https://github.com/QwenLM)** —— Qwen3.6 / Qwen3.8 模型家族（Apache-2.0）。

## 许可

本包为 MIT。引擎二进制为 Apache-2.0，FFmpeg / libcurl / zlib 以独立 DLL 形式与它同目录分发 ——
见 [LICENSE](LICENSE) 与 [NOTICE.md](NOTICE.md)。
