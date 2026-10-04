# tyrian-serve

OpenAI-compatible inference server for the Tyrian models (custom `TyrianForCausalLM`, loaded with `trust_remote_code=True`), running on CUDA (falls back to CPU). Defaults to **tyrian-500m**; tyrian-75m still works (see below). Useful as a custom model endpoint for Open WebUI.

Context length defaults to the length the model was trained on (`max_seq_len` in its `config.json`: **8192** for the 500M, 2048 for the 75M). Set `NUM_CTX` to override; RoPE buffers are recomputed at startup, so larger values work but are extrapolation and quality degrades.

## Setup

1. Create a virtualenv and install dependencies (the torch wheel is the CUDA 13.0 build):

   ```bash
   python3 -m venv .venv
   ./.venv/bin/pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu130
   ```

2. Get the model weights. They are not in this repo (`.modelcache/` is git-ignored). The server loads them from `.modelcache/tyrian-500m/` by default. The 500M is not on Hugging Face yet, so export it from its checkpoint and link it in:

   ```bash
   (cd ../tyrian-500m && python3 export_hf.py)          # writes ../tyrian-500m/hf_export/
   ln -s "$(realpath ../tyrian-500m/hf_export)" .modelcache/tyrian-500m
   ```

   To use a different directory, set `MODEL_PATH` (absolute, or relative to this repo).

   **tyrian-75m:** download it from <https://huggingface.co/redptam/tyrian-75m> and run with `MODEL_PATH` and `MODEL_NAME` pointing at it:

   ```bash
   ./.venv/bin/hf download redptam/tyrian-75m --local-dir .modelcache
   MODEL_PATH=.modelcache MODEL_NAME=tyrian-75m ./.venv/bin/python -m uvicorn server:app --host 0.0.0.0 --port 8000
   ```

## Run

From this directory:

```bash
./.venv/bin/python -m uvicorn server:app --host 0.0.0.0 --port 8000
```

Env vars:
- `NUM_CTX` — context window (default: the model's trained length, `8192` for the 500M)
- `DEVICE` — `auto` (default) / `cuda` / `cpu`
- `CUDA_DEVICE` — GPU index (default `1`)
- `MODEL_NAME` — id exposed at `/v1/models` and echoed back (default `tyrian-500m`)
- `MODEL_PATH` — weights directory, absolute or relative to this repo (default `.modelcache/tyrian-500m`)
- `PORT`, `HOST` — bind address for the built-in runner

Endpoints:
- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions` — OpenAI format; supports `stream:true` (SSE) and:
  - sampling: `temperature`, `top_p`, `top_k`
  - length: `max_completion_tokens` (preferred) or `max_tokens` (default 256)
  - `stop`: string or list; output is cut before the first match (never partially streamed)
  - penalties over generated tokens: `frequency_penalty`, `presence_penalty` (OpenAI semantics) and `repetition_penalty` / `repeat_penalty` (multiplicative, 1.0 = off). The 75M loops badly without one; `repeat_penalty` ≈ 1.2–1.3 works well.
- Generation stops at `<|im_end|>` (end of the assistant turn), `<eos>` or `<pad>`.

Quick check:

```bash
curl -s http://localhost:8000/v1/models | python3 -m json.tool
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"tyrian-500m","messages":[{"role":"user","content":"Say hello"}],"max_tokens":64}' | python3 -m json.tool
```

## Open WebUI wiring

Open **Administration → Connections (API)** in Open WebUI:

1. Endpoint / Base URL: `http://<this-host>:8000/v1`  (use your machine's LAN IP or hostname; port = whatever you started the server on)
2. Model Name: `tyrian-500m` (must match `MODEL_NAME`, which is what `/v1/models` lists)
3. API Key: leave empty or any dummy string (the server does not check one)

Then create a model entry with the same name and set **context size to 8192** in its settings (2048 for the 75M). On an RTX 5060 Ti the 500M generates roughly 60 tokens/s (the 75M roughly 200).

## Notes / limitations

- A lock serializes each decode step (not whole requests), so a client that disconnects mid-stream can't wedge the server; concurrent requests interleave.
- Streaming generation runs in a worker thread, so `/health` and new requests stay responsive.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set by default: each step allocates slightly larger tensors, and without it the caching allocator fragments into several GB (OOM when sharing the GPU).
- KV cache (`forward_cached` in `server.py`): the prompt is processed once, then each step runs only the new token against cached keys/values. It is preallocated per request for prompt + `max_tokens`: up to ~512 MB for the 500M at a full 8192 ctx (~17 MB for the 75M at 2048). It reuses the model's own layers; if `modeling_tyrian.py` changes, keep it in sync. Throughput is ~60 tok/s for the 500M (32 layers) and ~200 tok/s for the 75M (8 layers), flat with context length; each step is bound by Python/kernel-launch overhead rather than GPU compute, so CUDA graphs are the next speedup.
- These are small, lightly-trained LMs: treat outputs as uncurated samples.
- Streaming chunks are decoded per token; rare multi-byte BPE splits can briefly show up until the next chunk completes them (self-corrects).
