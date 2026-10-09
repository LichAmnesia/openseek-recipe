# Client configs

Replace `YOUR_GPU_HOST` with the server's hostname or IP. The model id must match
what `GET /v1/models` returns (`qwen38-<QUANT>`, default `qwen38-exl3-ple8`).

Model compute happens on the GPU box. File reads, edits and shell commands happen on
whatever machine runs the client, in its working directory.

## OpenCode

```bash
mkdir -p ~/.config/opencode
cp clients/opencode.json ~/.config/opencode/opencode.rtx6000.json
sed -i.bak 's/YOUR_GPU_HOST/192.168.1.50/' ~/.config/opencode/opencode.rtx6000.json

OPENCODE_CONFIG=~/.config/opencode/opencode.rtx6000.json opencode -m rtx6000/qwen38-exl3-ple8
```

`OPENCODE_CONFIG` layers on top of your global config rather than replacing it, so this
leaves your other providers alone. Uses Chat Completions (`/v1/chat/completions`).

## Codex CLI

```bash
cp clients/codex.rtx6000.config.toml ~/.codex/rtx6000.config.toml
sed -i.bak 's/YOUR_GPU_HOST/192.168.1.50/' ~/.codex/rtx6000.config.toml

codex -p rtx6000 -C /path/to/your/project
```

Passing only `-m qwen38-exl3-ple8` is not enough; the model id doesn't carry the server
address. Codex uses the Responses API (`/v1/responses`), which this vLLM build serves.

Two warnings are expected and harmless:

```text
Model metadata for `qwen38-exl3-ple8` not found. Defaulting to fallback metadata
failed to decode models response: missing field `models`
```

vLLM's `/v1/models` returns an OpenAI-style `data` array, not Codex's own catalog
format. Requests and tool calls still work.

## Anything OpenAI-compatible

```bash
export OPENAI_BASE_URL=http://YOUR_GPU_HOST:8001/v1
export OPENAI_API_KEY=not-needed
```

To turn thinking off for a single request (faster, fewer tokens):

```json
{"chat_template_kwargs": {"enable_thinking": false}}
```

## Verify from the client machine

```bash
python3 scripts/smoke_test.py --base-url http://YOUR_GPU_HOST:8001
```

Checks `/v1/models`, plain chat, reasoning output, a real tool call, the Responses API
and streaming. A client exiting 0 or a model saying "done" is not proof tool calling
works; this script checks the actual `tool_calls` payload.
