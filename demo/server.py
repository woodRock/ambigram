from __future__ import annotations

import asyncio
import logging
import sys
import threading
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.char_classifier import CharClassifier
from src.svg_optimizer import OptimUpdate, SVGAmbigramOptimizer

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------

_STATIC   = Path(__file__).parent / "static"
_CKPT     = Path("data/char_classifier.pth")

app = FastAPI(title="Ambigram Generator")
app.mount("/static", StaticFiles(directory=_STATIC), name="static")


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE:     torch.device              = _device()
CLASSIFIER: CharClassifier | None    = None


@app.on_event("startup")
async def _load_models() -> None:
    global CLASSIFIER
    log.info("Device: %s", DEVICE)
    if _CKPT.exists():
        CLASSIFIER = CharClassifier.load(_CKPT, DEVICE)
        log.info("Classifier ready.")
    else:
        log.warning(
            "No classifier at %s — run:\n"
            "  python tools/generate_font_dataset.py\n"
            "  python tools/train_classifier.py",
            _CKPT,
        )


@app.get("/")
async def _index() -> HTMLResponse:
    return HTMLResponse((_STATIC / "index.html").read_text())


@app.websocket("/ws")
async def _ws(websocket: WebSocket) -> None:
    await websocket.accept()

    if CLASSIFIER is None:
        await websocket.send_json({
            "error": (
                "Classifier not loaded. "
                "Run: python tools/generate_font_dataset.py "
                "&& python tools/train_classifier.py"
            )
        })
        await websocket.close()
        return

    try:
        data = await websocket.receive_json()
    except Exception:
        await websocket.close()
        return

    word = data.get("word", "SWIMS").upper().strip()
    if not word.isalpha():
        await websocket.send_json({"error": "Word must contain only letters (A–Z)."})
        await websocket.close()
        return

    log.info("Starting optimisation: %s", word)

    loop:  asyncio.AbstractEventLoop               = asyncio.get_event_loop()
    queue: asyncio.Queue[OptimUpdate | None]       = asyncio.Queue()

    def _on_update(upd: OptimUpdate) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, upd)

    def _run() -> None:
        try:
            SVGAmbigramOptimizer(
                word=word,
                classifier=CLASSIFIER,  # type: ignore[arg-type]
                device=DEVICE,
                on_update=_on_update,
            ).run()
        except Exception as exc:
            log.exception("Optimisation failed: %s", exc)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    threading.Thread(target=_run, daemon=True).start()

    try:
        while True:
            upd = await asyncio.wait_for(queue.get(), timeout=300.0)
            if upd is None:
                break
            await websocket.send_json({
                "step":    upd.step,
                "phase":   upd.phase,
                "score":   round(upd.score * 100, 1),
                "pct":     round(upd.pct, 1),
                "svg":     upd.svg,
                "svg_rot": upd.svg_rot,
            })
    except (WebSocketDisconnect, asyncio.TimeoutError):
        log.info("Connection closed for: %s", word)


if __name__ == "__main__":
    uvicorn.run("demo.server:app", host="127.0.0.1", port=8000, reload=False)
