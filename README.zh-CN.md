# 单张 RTX PRO 6000 跑 Qwen3.8-Flash-Next

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

## 五步跑起来

```bash
git clone https://github.com/LichAmnesia/qwen38-flash-next-rtx6000.git
cd qwen38-flash-next-rtx6000 && cp .env.example .env
scripts/doctor.sh            # 只读体检：GPU、内存、磁盘、docker、端口
scripts/install-prereqs.sh   # 体检报 docker / nvidia runtime 缺失时才需要跑
scripts/setup.sh             # 固定版本的 recipe + 镜像 ~29GB + KV offload 镜像 + 权重 ~120GB
scripts/ctl.sh start         # 起 LMCache + vLLM 两个容器，等 /v1/models 就绪（约 3–5 分钟）
python3 scripts/smoke_test.py   # 依次验证 models、chat、思考模式、真实工具调用、Responses API、流式输出
```

默认开启 **KV cache 内存层**：一个独立的 LMCache 容器在主机内存里留 64 GiB，长会话被挤出 GPU 的 KV 池后从内存拉回，不用整段重算。需要 128GB 内存；原来只用 GPU 的版本放在 [legacy/no-offload](legacy/no-offload/README.md)。原理、内存怎么算、已知限制见 [docs/kv-ram-offload.md](docs/kv-ram-offload.md)。

端点是 `http://<GPU 主机>:8001/v1`，模型 id 为 `qwen38-exl3-ple8`，不需要 key。客户端配置见 [clients/README.md](clients/README.md)。

## 几个上游 README 没写、要自己踩才知道的点

1. **权重实际是 120GB，不是 48GiB**：48GiB 只是 PLE 表本身的大小，磁盘至少留 150GB。
2. **Max-Q 少了 25% 功耗，速度只掉约 10%**：C1 加权掉 9%，C16 掉 9.8%。
3. **Max-Q 上的 MoE kernel 还没调过**：B12x 找不到 Max-Q 的 profile，退回启发式配置。芯片和 SM 数跟 400W 版一样，还有提升空间。
4. **只报 tok/s 不说负载，意义不大**：MTP 接受率从 52% 到 97% 不等，同一个服务在多语言散文上是 102 tok/s，在重复性输出上能到 215 tok/s。
5. **SWE-bench Pro 的 harness 有三个静默失败的坑**，其中一个白烧了 186 次模型调用才暴露。见 [docs/swebench-pro.md](docs/swebench-pro.md)。
6. **KV 内存层比 GPU 池小就等于没有**：LMCache 的内存层会把 GPU 里已有的也存一份，32 GiB（约 60 万 token）对上约 80 万 token 的 GPU 池，实测几乎零命中。64 GiB（约 125 万 token）让 9 个会话轮转从每轮 142 s 降到 10.7 s，decode 速度不变。
7. **vLLM 自带的 offload 和老版 LMCache 都过不了这个模型**：QSA 有一个不能缓存的临时环形缓冲区，要 LMCache ≥ 0.5.6 的 MP connector 才会跳过它。

## 致谢

真正难的工程在 [tpurtell/sm12x-exl3-qwen3.8-flash-next](https://github.com/tpurtell/sm12x-exl3-qwen3.8-flash-next)（Apache-2.0）。本仓库做的是：钉住它的版本，补上体检、生命周期管理、冒烟测试、客户端配置，以及第一份 Max-Q 300W 的实测数据。权重来自 [wrldsuksgo2mars](https://huggingface.co/wrldsuksgo2mars/Qwen3.8-Flash-Next-EXL3-K4.25-PLE-FP8-v1)，模型来自 [Qwen](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)。

本仓库不分发任何模型权重或镜像，`setup.sh` 会从原始来源下载。Qwen 模型和 EXL3 量化权重在 Hugging Face 上的许可都是 `license: other`，使用前请自行阅读并接受，商用前尤其要看清楚。
