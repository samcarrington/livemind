#!/usr/bin/env python3
"""
Local STT server for LiveMind.

Replaces the tunnelled faster-whisper service (stt.btrbot.com) with a
process on this machine. Implements the interface stt_worker.py expects:

  GET  /health                 → {"status": "ok", "model": "...", "engine": "..."}
  POST /v1/transcribe?sample_rate=16000&language=en&initial_prompt=...
       body: raw little-endian float32 mono PCM
       → {"text": "...", "language": "en", "processing_s": 0.42}

Engines:
  mlx            mlx-whisper (Apple Silicon GPU, fastest on a Mac)
  faster-whisper CTranslate2 (CPU int8, or CUDA on Linux/NVIDIA)
  auto (default) mlx if installed on Apple Silicon, else faster-whisper

Usage:
  uv sync --extra stt-mlx        # or: --extra stt
  uv run python stt_server.py                     # :8766, auto engine
  uv run python stt_server.py --engine faster-whisper --model small
  uv run python stt_server.py --model mlx-community/whisper-large-v3-turbo
"""

import argparse, os, platform, sys, threading, time

import numpy as np
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# Load .env (same file app.py uses) so STT_* settings live in one place
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

DEFAULT_MODELS = {
    "mlx": "mlx-community/whisper-large-v3-turbo",
    "faster-whisper": "small",
}


def _mlx_available() -> bool:
    if sys.platform != "darwin" or platform.machine() != "arm64":
        return False
    try:
        import mlx_whisper  # noqa: F401
        return True
    except ImportError:
        return False


class Engine:
    name = ""
    model = ""

    def transcribe(self, audio: np.ndarray, language: str | None, prompt: str | None) -> tuple[str, str]:
        raise NotImplementedError


class MLXEngine(Engine):
    name = "mlx"

    def __init__(self, model: str):
        import mlx_whisper
        self._mlx = mlx_whisper
        self.model = model
        # Warm up: downloads weights on first run and compiles kernels
        print(f"  Loading mlx-whisper model {model} (first run downloads weights)...")
        self._mlx.transcribe(np.zeros(16000, dtype=np.float32), path_or_hf_repo=model)

    def transcribe(self, audio, language, prompt):
        result = self._mlx.transcribe(
            audio,
            path_or_hf_repo=self.model,
            language=language or None,
            initial_prompt=prompt or None,
            condition_on_previous_text=False,
            temperature=0.0,
        )
        return (result.get("text") or "").strip(), result.get("language") or (language or "")


class FasterWhisperEngine(Engine):
    name = "faster-whisper"

    def __init__(self, model: str, device: str, compute_type: str):
        from faster_whisper import WhisperModel
        self.model = model
        print(f"  Loading faster-whisper model {model} ({device}/{compute_type})...")
        self._m = WhisperModel(model, device=device, compute_type=compute_type)

    def transcribe(self, audio, language, prompt):
        segments, info = self._m.transcribe(
            audio,
            language=language or None,
            initial_prompt=prompt or None,
            beam_size=5,
            vad_filter=True,
            condition_on_previous_text=False,
        )
        text = " ".join(s.text.strip() for s in segments).strip()
        return text, info.language or (language or "")


def build_engine(args) -> Engine:
    engine = args.engine
    if engine == "auto":
        engine = "mlx" if _mlx_available() else "faster-whisper"
    model = args.model or DEFAULT_MODELS[engine]
    if engine == "mlx":
        return MLXEngine(model)
    return FasterWhisperEngine(model, args.device, args.compute_type)


def create_app(engine: Engine) -> FastAPI:
    app = FastAPI(title="LiveMind local STT")
    lock = threading.Lock()  # models aren't safe for concurrent inference

    @app.get("/health")
    def health():
        return {"status": "ok", "model": engine.model, "engine": engine.name}

    @app.post("/v1/transcribe")
    async def transcribe(request: Request, sample_rate: int = 16000,
                         language: str = "", initial_prompt: str = ""):
        body = await request.body()
        if not body or len(body) % 4:
            return JSONResponse({"error": "body must be float32 PCM"}, status_code=400)
        audio = np.frombuffer(body, dtype=np.float32).copy()
        if sample_rate != 16000:
            n_out = max(1, int(len(audio) * 16000 / sample_rate))
            audio = np.interp(np.linspace(0, len(audio) - 1, n_out),
                              np.arange(len(audio)), audio).astype(np.float32)

        def run():
            with lock:
                t0 = time.time()
                text, lang = engine.transcribe(audio, language, initial_prompt)
                return text, lang, time.time() - t0

        import asyncio
        text, lang, dt = await asyncio.get_running_loop().run_in_executor(None, run)
        print(f"  [{engine.name}] {len(audio)/16000:.1f}s audio → {dt:.2f}s, lang={lang}: {text[:80]!r}")
        return {"text": text, "language": lang, "processing_s": round(dt, 3)}

    return app


def main():
    p = argparse.ArgumentParser(description="LiveMind local STT server")
    p.add_argument("--host", default=os.environ.get("STT_HOST") or "127.0.0.1")
    p.add_argument("--port", type=int, default=int(os.environ.get("STT_PORT") or 8766))
    p.add_argument("--engine", choices=["auto", "mlx", "faster-whisper"],
                   default=os.environ.get("STT_ENGINE") or "auto")
    p.add_argument("--model", default=os.environ.get("STT_MODEL", ""),
                   help="Model name/repo (default depends on engine)")
    p.add_argument("--device", default=os.environ.get("STT_DEVICE") or "auto",
                   help="faster-whisper device: auto|cpu|cuda")
    p.add_argument("--compute-type", default=os.environ.get("STT_COMPUTE_TYPE") or "int8",
                   help="faster-whisper compute type: int8|float16|...")
    args = p.parse_args()

    engine = build_engine(args)
    print(f"  STT ready: engine={engine.name} model={engine.model} on http://{args.host}:{args.port}")
    uvicorn.run(create_app(engine), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
