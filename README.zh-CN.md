# OpenSeek Recipe

可复现的本地模型部署：固定权重与 runtime 版本、硬件体检、服务管理、客户端接入和原始评测结果。

这个仓库负责 **recipe**。聊天和 coding 使用已有 **OpenSeek TUI**；性能与监控后续联动
**SparkDash**。统一入口是仓库内的 `./recipe`，不会替换现有 `openseek` 命令。

首个已验证 recipe：**Qwen3.8-Flash-Next + 单张 RTX PRO 6000 Blackwell Max-Q 96GB**。
新增 [DeepSeek-V4.1-Flash-1017 2.7bpw 实验版跑法](docs/deepseek-v41-flash-1017.md)：
96GB 显存 +128GB 内存，TP1/eager 实测 **10.64 tok/s、TTFT 5.31 秒**。
包含固定版本、下载/校验、混合位宽与磁盘 Engram、MXFP8 输出层补丁、启动与视频服务交接。
尚未通过速度门槛，KL/SWE-bench/256K 质量未验收；通过独立实验入口运行。
其余配置与命令见 [CLI 指南](docs/CLI.md)。

DeepSeek 权重、固定跑法快照与原始实测数据已发布至
[OpenSeek EXL3 Hugging Face 仓库](https://huggingface.co/LichAmnesia/DeepSeek-V4.1-Flash-1017-OpenSeek-EXL3)
（`v0.1.0-eager`，保留 sfxnz 原权重与署名，运行栈仍为实验版）。

仓库原名 `qwen38-flash-next-rtx6000`。

## 单张 RTX PRO 6000 跑 Qwen3.8-Flash-Next

用一张 **RTX PRO 6000 Blackwell 96GB**，通过 vLLM 跑 **Qwen3.8-Flash-Next**（125B MoE / 6B 激活 / 262K 上下文 / 视觉 / 工具调用）。跑起来就是一个 OpenAI 兼容端点，OpenCode、Codex CLI 等 coding agent 可以直接接。

**以下全部是在 300W Max-Q 卡上实测的数据：**

| | |
|---|---|
| 单并发解码（code / math） | **181 / 186 tok/s** |
| 七类负载加权解码 | **134.4 tok/s**（比 400W 参考低 9%） |
| 16 并发总吞吐 | **595 tok/s** |
| 首 token 延迟 | **85–102 ms** |
| SWE-bench Pro（50 题分层抽样） | **66.0%**（33/50）；Qwen 官方卡标的是 62.5 |
| 工具调用 / 视觉 | 16/16 / 1、4、16 图全过 |
| 显存 / 功耗 | 92.8 GB / 300 W |
| 长会话被挤出 GPU 后从内存层拉回 | **0.64 s**（不用再花 19.5 s 重算 130K token） |

## Clone 后统一使用 recipe

```bash
git clone https://github.com/LichAmnesia/openseek-recipe.git
cd openseek-recipe
./recipe models list
./recipe models select qwen38-flash-next  # 只保存本地选择，不切换正在运行的服务
./recipe install --dry-run               # 预览配置和安装步骤
./recipe install                         # 体检、固定镜像、断点下载、完整 SHA256 校验
./recipe start                           # LMCache → vLLM，等待 API
./recipe smoke                           # 真实工具调用、推理、Responses、流式等六项检查
./recipe connection                      # 输出给 OpenSeek TUI 使用的 endpoint/model
```

需要宿主 Python 3.10+；CLI 本身不依赖第三方 Python 包。体检提示缺 Docker/NVIDIA runtime 时，
先运行 `scripts/install-prereqs.sh`，再重试。当前登记一个已验证 Qwen 配置；未来自己的 HF 发布
完成制品和 loader 验证后再登记，不提前放不存在的模型。

选择保存于忽略提交的 `.recipe/config.json`；权重 40 个文件的 revision/大小/SHA256、
runtime commit、基础镜像 digest 和 LMCache wheel hash 固定于 `locks/`。
完整快照为 **128.33 GB（十进制，约 119.5 GiB）**，包含 22 个权重分片。
重复安装复用完整缓存，但会重新计算 hash，可能耗时数分钟。
配置优先级、命令和注册新模型见 [docs/CLI.md](docs/CLI.md)。

已有旧服务时先安装新锁定镜像，再自行执行 `./recipe recreate` 切换；`start/restart` 遇到
旧镜像会明确报版本不一致。修改配置也需 `recreate`，保存模型选择本身不影响正在运行的服务。

默认开启 **KV cache 内存层**：一个独立的 LMCache 容器在主机内存里留 64 GiB，长会话被挤出 GPU 的 KV 池后从内存拉回，不用整段重算。需要 128GB 内存；原来只用 GPU 的版本放在 [legacy/no-offload](legacy/no-offload/README.md)。原理、内存怎么算、已知限制见 [docs/kv-ram-offload.md](docs/kv-ram-offload.md)。

端点是 `http://<GPU 主机>:8001/v1`，模型 id 为 `qwen38-exl3-ple8`，不需要 key。客户端配置见 [clients/README.md](clients/README.md)。

## 几个上游 README 没写、要自己踩才知道的点

1. **完整快照为 119.5 GiB（128.33 GB），不是 48 GiB**：48 GiB 只是 PLE 表本身，连同镜像和构建预留建议留约 160 GiB。
2. **Max-Q 少了 25% 功耗，速度只掉约 10%**：C1 加权掉 9%，C16 掉 9.8%。
3. **Max-Q 上的 MoE kernel 还没调过**：B12x 找不到 Max-Q 的 profile，退回启发式配置。芯片和 SM 数跟 400W 版一样，还有提升空间。
4. **只报 tok/s 不说负载，意义不大**：MTP 接受率从 52% 到 97% 不等，同一个服务在多语言散文上是 102 tok/s，在重复性输出上能到 215 tok/s。
5. **SWE-bench Pro 的 harness 有三个静默失败的坑**，其中一个白烧了 186 次模型调用才暴露。见 [docs/swebench-pro.md](docs/swebench-pro.md)。
6. **KV 内存层比 GPU 池小就等于没有**：LMCache 的内存层会把 GPU 里已有的也存一份，32 GiB（约 60 万 token）对上约 80 万 token 的 GPU 池，实测几乎零命中。64 GiB（约 125 万 token）让 9 个会话轮转从每轮 142 s 降到 10.7 s，decode 速度不变。
7. **vLLM 自带的 offload 和老版 LMCache 都过不了这个模型**：QSA 有一个不能缓存的临时环形缓冲区，要 LMCache ≥ 0.5.6 的 MP connector 才会跳过它。

## 致谢

真正难的工程在 [tpurtell/sm12x-exl3-qwen3.8-flash-next](https://github.com/tpurtell/sm12x-exl3-qwen3.8-flash-next)（Apache-2.0）。本仓库做的是：钉住它的版本，补上体检、生命周期管理、冒烟测试、客户端配置，以及第一份 Max-Q 300W 的实测数据。权重来自 [wrldsuksgo2mars](https://huggingface.co/wrldsuksgo2mars/Qwen3.8-Flash-Next-EXL3-K4.25-PLE-FP8-v1)，模型来自 [Qwen](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)。

本仓库不分发任何模型权重或镜像，`setup.sh` 会从原始来源下载。Qwen 模型和 EXL3 量化权重在 Hugging Face 上的许可都是 `license: other`，使用前请自行阅读并接受，商用前尤其要看清楚。

DeepSeek 实验权重来自 [sfxnz](https://huggingface.co/sfxnz/DeepSeek-V4.1-Flash-1017-EXL3)，
host 专家底座来自 [Diffbot](https://huggingface.co/diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000)。
编排脚本采用 MIT；混合位宽插件补丁及生成的插件修改采用 **AGPL-3.0-only**，
详见 [来源与许可说明](experiments/deepseek-v41-flash-1017/THIRD_PARTY_NOTICES.md)。
