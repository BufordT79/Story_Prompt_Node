"""Thin wrapper around llama-cpp-python with a resident model cache.

The loaded GGUF model stays in memory across node executions and is only
reloaded when the model path or its init parameters (context length, GPU
layers) change — mirroring the artfat-style approach so repeated queue runs
don't pay the model-load cost every time.
"""

import gc
import os
import re

_LOG_PREFIX = "[StoryPromptBatch]"

_THINK_CLOSE = re.compile(r"(?i)</think(?:ing)?>")
_THINK_OPEN = re.compile(r"(?is)<think(?:ing)?>.*$")


def strip_reasoning(text):
    """Drop chain-of-thought from thinking models.

    Handles closed <think>...</think> blocks, a bare closing tag with no
    opener (some chat templates auto-open the block), and an unclosed opener
    (reasoning truncated by max_tokens — nothing usable follows it).
    """
    if _THINK_CLOSE.search(text):
        text = _THINK_CLOSE.split(text)[-1]
    return _THINK_OPEN.sub(" ", text).strip()

_cache = {"key": None, "model": None}


def _import_llama():
    try:
        from llama_cpp import Llama
        return Llama
    except ImportError as exc:
        raise RuntimeError(
            "llama-cpp-python is not installed in ComfyUI's Python environment.\n"
            "Standard install:   pip install -r ComfyUI/custom_nodes/Story_Prompts_Node/requirements.txt\n"
            "Windows portable:   .\\python_embeded\\python.exe -m pip install -r "
            ".\\ComfyUI\\custom_nodes\\Story_Prompts_Node\\requirements.txt\n"
            "For GPU offload you need a CUDA/Metal build of llama-cpp-python — see the README."
        ) from exc


def free_model():
    """Release the cached model (called automatically before loading a new one)."""
    model = _cache["model"]
    _cache["model"] = None
    _cache["key"] = None
    if model is not None:
        try:
            model.close()
        except Exception:
            pass
        del model
        gc.collect()


def get_model(model_path, n_ctx, n_gpu_layers):
    """Return the cached Llama instance, loading/reloading only when needed."""
    Llama = _import_llama()
    key = (os.path.abspath(model_path), int(n_ctx), int(n_gpu_layers))
    if _cache["key"] == key and _cache["model"] is not None:
        return _cache["model"]
    free_model()
    print(f"{_LOG_PREFIX} loading GGUF model: {os.path.basename(model_path)} "
          f"(n_ctx={n_ctx}, n_gpu_layers={n_gpu_layers})")
    try:
        model = Llama(
            model_path=model_path,
            n_ctx=int(n_ctx),
            n_gpu_layers=int(n_gpu_layers),
            verbose=False,
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to load GGUF model '{model_path}': {exc}") from exc
    _cache["key"] = key
    _cache["model"] = model
    return model


def count_tokens(model, text):
    try:
        return len(model.tokenize(text.encode("utf-8"), add_bos=False))
    except Exception:
        return max(1, len(text) // 4)  # rough 4-chars-per-token estimate


def chat(model, messages, *, max_tokens, seed, temperature, top_p, top_k, repeat_penalty):
    """One chat completion; returns the stripped reply text.

    The per-call `seed` keyword only exists in newer llama-cpp-python versions,
    hence the TypeError fallback.
    """
    kwargs = dict(
        messages=messages,
        max_tokens=int(max_tokens),
        temperature=float(temperature),
        top_p=float(top_p),
        top_k=int(top_k),
        repeat_penalty=float(repeat_penalty),
    )
    try:
        out = model.create_chat_completion(seed=int(seed) & 0x7FFFFFFF, **kwargs)
    except TypeError:
        out = model.create_chat_completion(**kwargs)
    content = out["choices"][0]["message"].get("content") or ""
    return strip_reasoning(content.strip())
