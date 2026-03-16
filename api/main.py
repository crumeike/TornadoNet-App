"""
TornadoNet FastAPI Backend
CRADA CN-24-0590 | JHU · U of Alabama · U of South Alabama · NIST

Loads TornadoNet .pt checkpoints via Ultralytics and exposes:
  POST /predict         — single image inference
  POST /predict/batch   — multiple images
  GET  /health          — connection test + loaded models
  GET  /models          — available model metadata
"""

import io
import base64
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from ultralytics import YOLO, RTDETR
from PIL import Image

# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="TornadoNet Inference API",
    description="Post-tornado building damage detection using IN-CORE DS0–DS4 classification",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── IN-CORE damage state metadata ─────────────────────────────────────────────

DS_LABELS = {
    0: {"id": "DS0", "name": "Undamaged",  "color": "#4a9e6b"},
    1: {"id": "DS1", "name": "Slight",     "color": "#e6b800"},
    2: {"id": "DS2", "name": "Moderate",   "color": "#e8883a"},
    3: {"id": "DS3", "name": "Extensive",  "color": "#d44e2a"},
    4: {"id": "DS4", "name": "Complete",   "color": "#8b1a1a"},
}

# ── Model registry ─────────────────────────────────────────────────────────────

CHECKPOINT_DIR = Path(__file__).parent.parent / "checkpoints"

MODEL_REGISTRY = {
    "rtdetr-ord": {
        "label":       "RT-DETR-L + Ordinal Supervision (psi=0.5, K=1)",
        "short":       "RT-DETR-L + Ordinal",
        "file":        "tornadonet-rtdetr-l-ordinal-psi0.5-k1.pt",
        "hf_repo":     "crumeike/tornadonet-checkpoints",
        "hf_filename": "tornadonet-rtdetr-l-ordinal-psi0.5-k1/best.pt",
        "map50":       44.70,
        "maoe":        0.56,
        "ord_acc":     91.15,
        "fps":         78,
        "params_m":    32.0,
        "loader":      "rtdetr",
    },
    "yolo11x": {
        "label":       "YOLO11x Baseline",
        "short":       "YOLO11x",
        "file":        "tornadonet-yolo11-x-baseline.pt",
        "hf_repo":     "crumeike/tornadonet-checkpoints",
        "hf_filename": "tornadonet-yolo11-x-baseline/best.pt",
        "map50":       46.05,
        "maoe":        0.76,
        "ord_acc":     85.20,
        "fps":         66,
        "params_m":    56.8,
        "loader":      "yolo",
    },
    "yolo11n": {
        "label":       "YOLO11n Baseline",
        "short":       "YOLO11n",
        "file":        "tornadonet-yolo11-n-baseline.pt",
        "hf_repo":     "crumeike/tornadonet-checkpoints",
        "hf_filename": "tornadonet-yolo11-n-baseline/best.pt",
        "map50":       41.14,
        "maoe":        0.77,
        "ord_acc":     84.79,
        "fps":         239,
        "params_m":    2.6,
        "loader":      "yolo",
    },
}

# Cache loaded models so we don't reload on every request
_model_cache: dict = {}


def load_model(model_id: str):
    """Load a model by registry ID, caching after first load."""
    if model_id in _model_cache:
        return _model_cache[model_id]

    if model_id not in MODEL_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not in registry.")

    meta = MODEL_REGISTRY[model_id]
    ckpt_path = CHECKPOINT_DIR / meta["file"]

    # Try local checkpoint first, fall back to HuggingFace
    if ckpt_path.exists():
        weights = str(ckpt_path)
    else:
        # Requires huggingface_hub installed and HF_TOKEN set if private
        try:
            from huggingface_hub import hf_hub_download
            weights = hf_hub_download(
                repo_id=meta["hf_repo"],
                filename=meta["hf_filename"],
            )
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Checkpoint not found locally at '{ckpt_path}' "
                    f"and HuggingFace download failed: {e}. "
                    f"Place the .pt file in checkpoints/ or set HF_TOKEN."
                ),
            )

    loader = RTDETR if meta["loader"] == "rtdetr" else YOLO
    model = loader(weights)
    _model_cache[model_id] = model
    return model


# ── Helpers ────────────────────────────────────────────────────────────────────

def bytes_to_cv2(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")
    return img


def cv2_to_base64(img: np.ndarray, quality: int = 90) -> str:
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def run_inference(model_id: str, img_bgr: np.ndarray, conf_threshold: float = 0.25):
    """Run inference and return structured detection results + annotated image."""
    model = load_model(model_id)
    t0 = time.perf_counter()
    results = model(img_bgr, conf=conf_threshold, verbose=False)
    inference_ms = round((time.perf_counter() - t0) * 1000, 1)

    detections = []
    for r in results:
        boxes = r.boxes
        if boxes is None:
            continue
        for box in boxes:
            cls_idx = int(box.cls[0])
            conf    = float(box.conf[0])
            x1, y1, x2, y2 = [round(float(v), 1) for v in box.xyxy[0]]
            ds = DS_LABELS.get(cls_idx, {"id": f"DS{cls_idx}", "name": "Unknown", "color": "#888888"})
            detections.append({
                "class_idx":   cls_idx,
                "ds_id":       ds["id"],
                "ds_name":     ds["name"],
                "ds_color":    ds["color"],
                "confidence":  round(conf, 4),
                "bbox_xyxy":   [x1, y1, x2, y2],
                "bbox_xywh":   [x1, y1, round(x2 - x1, 1), round(y2 - y1, 1)],
            })

    # Sort by severity descending, then confidence descending
    detections.sort(key=lambda d: (-d["class_idx"], -d["confidence"]))

    # Annotated image — Ultralytics renders boxes natively
    annotated = results[0].plot(line_width=2, font_size=10)
    annotated_b64 = cv2_to_base64(annotated)

    # Damage distribution summary
    dist = {f"DS{i}": 0 for i in range(5)}
    for d in detections:
        dist[d["ds_id"]] += 1

    return {
        "detections":       detections,
        "detection_count":  len(detections),
        "damage_distribution": dist,
        "annotated_image_b64": annotated_b64,
        "inference_ms":     inference_ms,
        "model_id":         model_id,
        "model_label":      MODEL_REGISTRY[model_id]["label"],
    }


# ── Pydantic models ────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    loaded_models: list[str]
    available_models: list[str]
    version: str


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Connection test. Returns loaded model cache and available model IDs."""
    return {
        "status":           "ok",
        "loaded_models":    list(_model_cache.keys()),
        "available_models": list(MODEL_REGISTRY.keys()),
        "version":          "1.0.0",
    }


@app.get("/models", tags=["System"])
async def list_models():
    """Return metadata for all available TornadoNet models."""
    return {
        mid: {k: v for k, v in meta.items() if k not in ("loader", "hf_filename")}
        for mid, meta in MODEL_REGISTRY.items()
    }


@app.post("/predict", tags=["Inference"])
async def predict(
    file: UploadFile = File(..., description="Street-view image (JPG, PNG)"),
    model_id: str = Query(default="rtdetr-ord", description="Model ID from /models"),
    conf: float  = Query(default=0.25, ge=0.01, le=0.99, description="Confidence threshold"),
):
    """
    Run TornadoNet inference on a single image.

    Returns:
    - `detections`: list of buildings with DS class, confidence, and bounding box
    - `annotated_image_b64`: JPEG annotated image as base64 string
    - `damage_distribution`: count per DS level
    - `inference_ms`: model forward pass time
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image (JPG or PNG).")

    raw = await file.read()
    img_bgr = bytes_to_cv2(raw)
    result = run_inference(model_id, img_bgr, conf_threshold=conf)
    return JSONResponse(content=result)


@app.post("/predict/batch", tags=["Inference"])
async def predict_batch(
    files: list[UploadFile] = File(..., description="Multiple street-view images"),
    model_id: str = Query(default="rtdetr-ord"),
    conf: float  = Query(default=0.25, ge=0.01, le=0.99),
):
    """
    Run TornadoNet inference on multiple images.

    Returns a list of per-image results in the same format as /predict,
    plus an aggregate damage distribution across all images.
    """
    if len(files) > 50:
        raise HTTPException(status_code=400, detail="Maximum 50 images per batch request.")

    results = []
    aggregate_dist = {f"DS{i}": 0 for i in range(5)}

    for f in files:
        if not f.content_type.startswith("image/"):
            continue
        raw = await f.read()
        img_bgr = bytes_to_cv2(raw)
        r = run_inference(model_id, img_bgr, conf_threshold=conf)
        r["filename"] = f.filename
        results.append(r)
        for ds_key, count in r["damage_distribution"].items():
            aggregate_dist[ds_key] += count

    total_buildings = sum(aggregate_dist.values())
    return JSONResponse(content={
        "image_count":       len(results),
        "total_buildings":   total_buildings,
        "aggregate_distribution": aggregate_dist,
        "results":           results,
        "model_id":          model_id,
        "model_label":       MODEL_REGISTRY[model_id]["label"],
    })


@app.post("/predict/video-frame", tags=["Inference"])
async def predict_video_frame(
    file: UploadFile = File(..., description="Video file (MP4, AVI, MOV)"),
    timestamp_sec: float = Query(default=0.0, ge=0.0, description="Timestamp in seconds to extract"),
    model_id: str = Query(default="rtdetr-ord"),
    conf: float  = Query(default=0.25, ge=0.01, le=0.99),
):
    """
    Extract a frame from a video at `timestamp_sec` and run inference.

    The frontend sends the selected frame timestamp from the frame strip.
    Returns the same structure as /predict plus the raw extracted frame as base64.
    """
    raw = await file.read()

    # Write to a temp file — OpenCV needs a file path for video
    import tempfile, os
    suffix = Path(file.filename).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(raw)
        tmp_path = tmp.name

    try:
        cap = cv2.VideoCapture(tmp_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_idx = int(timestamp_sec * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
    finally:
        os.unlink(tmp_path)

    if not ret or frame is None:
        raise HTTPException(status_code=400, detail=f"Could not extract frame at {timestamp_sec}s.")

    result = run_inference(model_id, frame, conf_threshold=conf)
    result["extracted_frame_b64"] = cv2_to_base64(frame)
    result["timestamp_sec"]       = timestamp_sec
    return JSONResponse(content=result)
