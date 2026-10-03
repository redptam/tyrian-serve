# tyrian-serve

OpenAI-compatible inference server for [redptam/tyrian-75m](https://huggingface.co/redptam/tyrian-75m) (custom `TyrianForCausalLM`, loaded with `trust_remote_code=True`), running on CUDA (falls back to CPU). Useful as a custom model endpoint for Open WebUI.

Context length defaults to **2048** (`NUM_CTX`), the length the model was trained on. RoPE buffers are recomputed at startup, so larger values work but are extrapolation and quality degrades.

## Setup

1. Create a virtualenv and install dependencies (the torch wheel is the CUDA 13.0 build):

   ```bash
   python3 -m venv .venv
   ./.venv/bin/pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu130
   ```

2. Download the model weights from Hugging Face: <https://huggingface.co/redptam/tyrian-75m>. They are not in this repo (`.modelcache/` is git-ignored). The server loads them from `.modelcache/` by default:

   ```bash
   ./.venv/bin/hf download redptam/tyrian-75m --local-dir .modelcache
   ```

   Or download the files manually from the model page into `.modelcache/`. To use a different directory, set `MODEL_PATH` (absolute, or relative to this repo).

## Run

From this directory:

```bash
./.venv/bin/python -m uvicorn server:app --host 0.0.0.0 --port 8000
```

Env vars:
- `NUM_CTX` — context window (default `2048`)
- `DEVICE` — `auto` (default) / `cuda` / `cpu`
- `CUDA_DEVICE` — GPU index (default `1`)
- `MODEL_NAME` — id exposed at `/v1/models` and echoed back (default `tyrian-75m`)
- `MODEL_PATH` — weights directory, absolute or relative to this repo (default `.modelcache`)
- `PORT`, `HOST` — bind address for the built-in runner

Endpoints:
- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions` — OpenAI format; supports `stream:true` (SSE) and:
  - sampling: `temperature`, `top_p`, `top_k`
  - length: `max_completion_tokens` (preferred) or `max_tokens` (default 256)
  - `stop`: string or list; output is cut before the first match (never partially streamed)
  - penalties over generated tokens: `frequency_penalty`, `presence_penalty` (OpenAI semantics) and `repetition_penalty` / `repeat_penalty` (multiplicative, 1.0 = off). This model loops badly without one; `repeat_penalty` ≈ 1.2–1.3 works well.

Quick check:

```bash
curl -s http://localhost:8000/v1/models | python3 -m json.tool
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"tyrian-75m","messages":[{"role":"user","content":"Say hello"}],"max_tokens":64}' | python3 -m json.tool
```

## Open WebUI wiring

Open **Administration → Connections (API)** in Open WebUI:

1. Endpoint / Base URL: `http://<this-host>:8000/v1`  (use your machine's LAN IP or hostname; port = whatever you started the server on)
2. Model Name: `tyrian-75m` (must match `MODEL_NAME`, which is what `/v1/models` lists)
3. API Key: leave empty or any dummy string (the server does not check one)

Then create a model entry with the same name and set **context size to 2048** in its settings. On an RTX 5060 Ti it generates roughly 150+ tokens/s.

## Notes / limitations

- A lock serializes each decode step (not whole requests), so a client that disconnects mid-stream can't wedge the server; concurrent requests interleave.
- Streaming generation runs in a worker thread, so `/health` and new requests stay responsive.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set by default: each step allocates slightly larger tensors, and without it the caching allocator fragments into several GB (OOM when sharing the GPU).
- KV cache (`forward_cached` in `server.py`): the prompt is processed once, then each step runs only the new token against cached keys/values (~17 MB per request at 2048 ctx). It reuses the model's own layers; if `.modelcache/modeling_tyrian.py` changes, keep it in sync. Throughput is ~200 tok/s and flat with context length; at this model size each step is now bound by Python/kernel-launch overhead (~220 launches, GPU busy ~1.5 of ~4.3 ms), so CUDA graphs are the next speedup.
- The model is a small, lightly-trained LM: treat outputs as uncurated samples.
- Streaming chunks are decoded per token; rare multi-byte BPE splits can briefly show up until the next chunk completes them (self-corrects).
