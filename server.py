import importlib
import json
import os
import threading
import time
import uuid
from typing import Iterator, List, Optional, Tuple, Union

# Each decode step allocates slightly larger tensors; without expandable segments the caching
# allocator fragments and reserves GBs for a ~0.5 GB working set (OOMs next to other GPU tenants).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import AliasChoices, BaseModel, Field

_HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_HERE, os.getenv("MODEL_PATH", ".modelcache/tyrian-500m"))
NUM_CTX = int(os.getenv("NUM_CTX", "0"))  # 0 = the model's trained context length (config.max_seq_len)
MODEL_ID = os.getenv("MODEL_NAME", "tyrian-500m")

from transformers import AutoTokenizer, AutoModelForCausalLM  # noqa: E402


def _pick_device_dtype():
    want = os.getenv("DEVICE", "auto").lower()

    def use_cuda(index: str):
        cap = torch.cuda.get_device_capability(int(index))
        name = torch.cuda.get_device_name(int(index))
        dt = torch.bfloat16 if cap[0] >= 8 else torch.float32
        print(f"tyrian-serve: backend=CUDA (torch {torch.__version__}) device=cuda:{index} ({name}, sm{cap[0]}{cap[1]}) dtype={dt}", flush=True)
        return f"cuda:{index}", dt

    if torch.cuda.is_available():
        explicit = os.getenv("CUDA_DEVICE", "1")
        if want in ("auto", "gpu", "cuda"):
            return use_cuda(explicit)
        print(f"tyrian-serve: CUDA is available but DEVICE={want!r}, staying on CPU (set DEVICE=auto to enable)", flush=True)
    elif torch.backends.mps.is_available() and want in ("auto", "mps"):
        print("tyrian-serve: backend=MPS (Apple Silicon); using float32, set DEVICE=cpu and expect a slower run here", flush=True)
        return "mps", torch.float32
    print(f"tyrian-serve: backend=CPU (torch {torch.__version__}; cuda_available={torch.cuda.is_available()}, mps={getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available()})", flush=True)
    return "cpu", torch.bfloat16


DEVICE, MODEL_DTYPE = _pick_device_dtype()


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = MODEL_ID
    messages: List[dict] = []
    temperature: float = 0.8
    top_p: Optional[float] = 1.0
    top_k: Optional[int] = 50
    max_tokens: Optional[int] = 256
    max_completion_tokens: Optional[int] = None  # newer OpenAI name; wins over max_tokens
    stop: Optional[Union[str, List[str]]] = None
    # Penalties apply to tokens generated so far (not the prompt, so the chat template isn't penalized).
    frequency_penalty: Optional[float] = None  # OpenAI: subtract penalty * count
    presence_penalty: Optional[float] = None  # OpenAI: subtract penalty once a token has appeared
    repetition_penalty: Optional[float] = Field(  # HF/llama.cpp style multiplicative, 1.0 = off
        default=None, validation_alias=AliasChoices("repetition_penalty", "repeat_penalty")
    )
    stream: bool = False


def load(path: str):
    global NUM_CTX
    print(f"loading model from {path}", flush=True)
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        path,
        trust_remote_code=True,
        dtype=MODEL_DTYPE,
    )

    trained_ctx = mdl.config.max_seq_len
    NUM_CTX = NUM_CTX or trained_ctx
    extrapolated = " (beyond trained length: extrapolation, quality degrades)" if NUM_CTX > trained_ctx else ""
    print(f"tyrian-serve: context window {NUM_CTX} tokens, trained on {trained_ctx}{extrapolated}", flush=True)

    mod = importlib.import_module(type(mdl).__module__)  # the custom module transformers loaded
    # Always rebuild RoPE: sizes it to NUM_CTX, and exports made before the _init_weights fix
    # (e.g. the published tyrian-75m) load these non-persistent buffers uninitialized
    cos, sin = mod.precompute_rope_freqs(
        mdl.config.hidden_size // mdl.config.num_heads, NUM_CTX, theta=mdl.config.rope_theta
    )
    mdl.to(DEVICE)
    mdl.register_buffer("rope_cos", cos.to(MODEL_DTYPE).to(DEVICE), persistent=False)
    mdl.register_buffer("rope_sin", sin.to(MODEL_DTYPE).to(DEVICE), persistent=False)
    mdl.config.max_seq_len = NUM_CTX
    tok.model_max_length = NUM_CTX

    # SFT ends each assistant turn with <|im_end|>, which the config's eos_token_id (<eos>) doesn't cover
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    stops = {
        v
        for v in (mdl.config.eos_token_id, getattr(mdl.config, "pad_token_id", None), im_end)
        if isinstance(v, int) and v != tok.unk_token_id
    }
    print(f"tyrian-serve: stop token ids {sorted(stops)}", flush=True)
    mdl.eval()
    mdl.requires_grad_(False)
    return tok, mdl, stops


print("tyrian-serve: loading model ...", flush=True)
TOK, MODEL, STOP_IDS = load(MODEL_PATH)
TYRIAN = importlib.import_module(type(MODEL).__module__)
GEN_LOCK = threading.Lock()

app = FastAPI(title="Tyrian Inference Server")


def sample_token(logits_row: torch.Tensor, temperature: float, top_p: Optional[float], top_k: Optional[int]) -> int:
    z = logits_row.float()
    if not (temperature > 0.0):
        return int(torch.argmax(z).item())

    z = z / max(float(temperature), 1e-5)
    vocab = z.shape[0]

    topk_int = int(top_k or 0)
    if 0 < topk_int < vocab:
        cutoff = torch.topk(z, topk_int).values[-1]
        z = z.masked_fill(z < cutoff, float("-inf"))

    if top_p is not None and 0.0 < float(top_p) < 1.0:
        sorted_z, sorted_idx = torch.sort(z, descending=True)
        sorted_p = torch.softmax(sorted_z, dim=0)
        # drop tokens once the mass *before* them already reaches top_p (always keeps the top token)
        drop_sorted = (sorted_p.cumsum(dim=0) - sorted_p) >= float(top_p)
        drop = torch.zeros_like(drop_sorted).scatter(0, sorted_idx, drop_sorted)
        z = z.masked_fill(drop, float("-inf"))

    probs = torch.softmax(z, dim=0)
    return int(torch.multinomial(probs, num_samples=1).item())


def apply_penalties(z: torch.Tensor, counts: torch.Tensor, req: "ChatCompletionRequest") -> torch.Tensor:
    if req.frequency_penalty:
        z = z - float(req.frequency_penalty) * counts
    if req.presence_penalty:
        z = z - float(req.presence_penalty) * (counts > 0)
    rp = req.repetition_penalty
    if rp and rp > 0 and rp != 1.0:
        penalized = torch.where(z > 0, z / rp, z * rp)
        z = torch.where(counts > 0, penalized, z)
    return z


class StopScanner:
    """Cuts output at the first stop string, holding back any tail that could still become one
    so a stop string is never partially streamed to the client."""

    def __init__(self, stop: Optional[Union[str, List[str]]]):
        stops = [stop] if isinstance(stop, str) else (stop or [])
        self.stops = [x for x in stops if x]
        self.buf = ""

    def feed(self, piece: str) -> Tuple[str, bool]:
        if not self.stops:
            return piece, False
        self.buf += piece
        hits = [i for i in (self.buf.find(x) for x in self.stops) if i >= 0]
        if hits:
            out, self.buf = self.buf[: min(hits)], ""
            return out, True
        hold = 0
        for x in self.stops:
            for k in range(min(len(x) - 1, len(self.buf)), hold, -1):
                if self.buf.endswith(x[:k]):
                    hold = k
                    break
        cut = len(self.buf) - hold
        out, self.buf = self.buf[:cut], self.buf[cut:]
        return out, False

    def flush(self) -> str:
        out, self.buf = self.buf, ""
        return out


def clean_piece(token_id: int) -> str:
    text = TOK.decode([token_id], skip_special_tokens=False)
    for token in ("<pad>", "<eos>", "<bos>"):
        text = text.replace(token, "")
    return text


def prompt_ids(messages: List[dict]):
    usable = [m for m in messages or [] if isinstance(m.get("content"), str)]
    if not usable:
        raise HTTPException(status_code=400, detail="messages must contain at least one string content")
    text = TOK.apply_chat_template(usable, tokenize=False, add_generation_prompt=True)
    ids = TOK(text, return_tensors="pt").input_ids.to(DEVICE)
    if ids.shape[1] >= NUM_CTX:
        raise HTTPException(status_code=400, detail=f"prompt exceeds context window ({NUM_CTX} tokens)")
    return ids


class KVCache:
    """Preallocated K/V buffers for every layer, one per request (~512 MB for the 500M at a full 8192 ctx)."""

    def __init__(self, max_len: int):
        cfg = MODEL.config
        head_dim = cfg.hidden_size // cfg.num_heads
        with torch.inference_mode():
            self.kv = torch.empty(
                (cfg.num_layers, 2, 1, cfg.num_kv_heads, max_len, head_dim), dtype=MODEL_DTYPE, device=DEVICE
            )
        self.len = 0


def forward_cached(ids: torch.Tensor, cache: KVCache) -> torch.Tensor:
    """Runs only the new tokens through the model, attending over the cached K/V, and returns the
    last position's logits. Mirrors TyrianForCausalLM.forward in .modelcache/modeling_tyrian.py."""
    B, T = ids.shape
    start, end = cache.len, cache.len + T
    # is_causal aligns its mask top-left, which is only right for prefill into an empty cache or a single token
    assert T == 1 or start == 0, "multi-token step only supported as the first (prefill) step"
    cos, sin = MODEL.rope_cos[start:end], MODEL.rope_sin[start:end]
    x = MODEL.embed_tokens(ids)
    for i, layer in enumerate(MODEL.layers):
        attn = layer.attn
        h = layer.attn_norm(x)
        q = attn.q_proj(h).view(B, T, attn.num_heads, attn.head_dim)
        k = attn.k_proj(h).view(B, T, attn.num_kv_heads, attn.head_dim)
        v = attn.v_proj(h).view(B, T, attn.num_kv_heads, attn.head_dim)
        q, k = TYRIAN.apply_rope(q, k, cos, sin)
        cache.kv[i, 0, :, :, start:end] = k.transpose(1, 2)
        cache.kv[i, 1, :, :, start:end] = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2), cache.kv[i, 0, :, :, :end], cache.kv[i, 1, :, :, :end], is_causal=T > 1, enable_gqa=True
        )
        x = x + attn.o_proj(out.transpose(1, 2).reshape(B, T, -1))
        x = x + layer.ffn(layer.ffn_norm(x))
    cache.len = end
    return MODEL.lm_head(MODEL.norm(x[:, -1]))[0]


def gen_loop(ids: torch.Tensor, max_tokens: int, req: ChatCompletionRequest) -> Iterator[Tuple[int, str]]:
    cache = KVCache(ids.shape[1] + max_tokens)
    step_ids = ids  # whole prompt on the first step, then just the last sampled token
    counts = torch.zeros(MODEL.config.vocab_size, device=ids.device)
    # Sync generator: Starlette iterates it in a worker thread, so it never blocks the event loop.
    # The lock and grad mode are scoped per step, never across a yield: a client that disconnects
    # leaves this generator suspended (not closed), which would otherwise hold the lock forever,
    # and grad mode is thread-local while each next() may run on a different worker thread.
    for _ in range(max_tokens):
        with GEN_LOCK, torch.inference_mode():
            row = forward_cached(step_ids, cache)
            row = apply_penalties(row.float(), counts, req)
            token_id = sample_token(row, req.temperature, req.top_p, req.top_k)
        counts[token_id] += 1
        yield int(token_id), clean_piece(token_id)
        step_ids = torch.tensor([[token_id]], dtype=ids.dtype, device=ids.device)
        if token_id in STOP_IDS:
            return


def generate_text(prompt: torch.Tensor, max_tokens: int, req: ChatCompletionRequest) -> Iterator[Tuple[str, int, Optional[str]]]:
    """Yields (text, tokens_so_far, finish_reason); finish_reason is set only on the last item."""
    scanner = StopScanner(req.stop)
    n = 0
    for token_id, piece in gen_loop(prompt, max_tokens, req):
        n += 1
        if token_id in STOP_IDS:
            yield scanner.flush(), n, "stop"
            return
        text, hit = scanner.feed(piece)
        if hit:
            yield text, n, "stop"
            return
        yield text, n, None
    yield scanner.flush(), n, "length"


def sse_event(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/v1/models")
def list_models():
    created = int(time.time())
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "created": created,
                "owned_by": "local",
            }
        ],
    }


@app.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionRequest):
    prompt = prompt_ids(request.messages)

    max_tokens = int(request.max_completion_tokens or request.max_tokens or 256)
    max_tokens = max(1, min(max_tokens, NUM_CTX - prompt.shape[1]))
    cid = "chatcmpl-" + uuid.uuid4().hex[:24]
    model_label = request.model or MODEL_ID
    created = int(time.time())

    if not request.stream:
        pieces: List[str] = []
        completion_tokens, finish_reason = 0, "length"
        for text, completion_tokens, finish in generate_text(prompt, max_tokens, request):
            pieces.append(text)
            finish_reason = finish or finish_reason

        content = "".join(pieces).strip()
        usage_prompt = int(prompt.shape[1])
        return {
            "id": cid,
            "object": "chat.completion",
            "created": created,
            "model": model_label,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": usage_prompt,
                "completion_tokens": completion_tokens,
                "total_tokens": usage_prompt + completion_tokens,
            },
        }

    def stream_response():
        first = True
        finish_reason = "length"
        for piece, _, finish in generate_text(prompt, max_tokens, request):
            finish_reason = finish or finish_reason
            if not piece:
                continue
            delta = {"role": "assistant", "content": piece} if first else {"content": piece}
            first = False
            yield sse_event(
                {
                    "id": cid,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_label,
                    "choices": [{"index": 0, "delta": delta}],
                }
            )

        yield sse_event(
            {
                "id": cid,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_label,
                "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
            }
        )
        yield "data: [DONE]\n\n"

    return StreamingResponse(stream_response(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", "8000")))
