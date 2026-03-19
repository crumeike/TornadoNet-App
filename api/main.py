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
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Form, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from ultralytics import YOLO, RTDETR
from PIL import Image

from dotenv import load_dotenv
load_dotenv()

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
    4: {"id": "DS0", "name": "Undamaged",  "color": "#4a9e6b"},
    0: {"id": "DS1", "name": "Slight",     "color": "#e6b800"},
    1: {"id": "DS2", "name": "Moderate",   "color": "#e8883a"},
    2: {"id": "DS3", "name": "Extensive",  "color": "#d44e2a"},
    3: {"id": "DS4", "name": "Complete",   "color": "#8b1a1a"},
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
    print(f"Running inference with model '{model_id}' at confidence threshold {conf_threshold}...")
    t0 = time.perf_counter()
    results = model(img_bgr, conf=conf_threshold, iou=0.5, agnostic_nms=True, verbose=False)
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

@app.post("/vlm", tags=["AI Analysis"])
async def vlm_assess(
    file: UploadFile = File(...),
    provider: str = Query(default="claude"),
    api_key: str = Form(default=""),  # ← Form instead of Query
):
    import httpx, base64

    # Read and encode image
    raw = await file.read()
    b64 = base64.standard_b64encode(raw).decode("utf-8")
    ext = file.content_type  # e.g. image/jpeg

    prompt = """You are a NIST structural engineer assessing post-tornado building damage from street-view imagery.
                Identify every visually distinct building you can assess in this image.
                For each building, apply the IN-CORE T1-T5 decision matrix (for wood frame archetypes T1-T5) and assign a damage state.

                Use the following IN-CORE T1-T5 damage state criteria:

                DS1 (Slight):     Roof covering 2-15% damaged AND/OR 1 window/door failure AND/OR no roof sheathing failure AND/OR no roof-to-wall connection failure
                DS2 (Moderate):   Roof covering 15-50% damaged AND/OR 2-3 windows/doors failed AND/OR 1-3 sections of roof sheathing failed AND/OR no roof-to-wall connection failure  
                DS3 (Extensive):  Roof covering >50% damaged AND/OR >3 windows/doors failed AND/OR >3 sheathing sections failed AND less than 35% sheathing area AND/OR no roof-to-wall connection failure
                DS4 (Complete):   Roof covering >50% damaged (typically) AND/OR >3 windows/doors failed (typically) AND/OR >35% of roof sheathing failed AND/OR roof-to-wall connection failure

                Rule: assign the rightmost (highest) DS column reached by ANY single element.
                DS0 (Undamaged): no visible damage to any element.

                For each building provide:
                - building: sequential number
                - location_description: relative position in the scene (e.g. "left foreground", "center background")
                - color: primary exterior color (e.g. "white", "red brick", "beige vinyl siding")
                - stories: number of visible stories (e.g. 1, 2, "1.5")
                - structure_type: brief description (e.g. "residential wood building", "light/heavy industrial building", "Office building", "business and retail building", 
                "community center/church", "hospital", "fire/police station", "mobile home", "brick ranch", "shopping center", "middle/high school (reinforced/Unreinforced masonry)")
                - plan_type: size and shape description (e.g. "small rectangular plan", " medium rectangular plan","large L-shaped plan")
                - roof_type: brief description (e.g. "gable roof", "hip roof", "flat roof", "roof covered with blue/black tarps", "partially collapsed roof")
                - roofCovering: observation and DS assignment
                - windowDoor: observation and DS assignment
                - roofSheathing: observation and DS assignment
                - roofWall: observation and DS assignment
                - overall: final DS0-DS4 using the rightmost column rule
                - confidence_pct: integer 0-100 representing your visual assessment confidence
                - rationale: 1 sentence referencing which element drove the final DS assignment

                Respond ONLY with a valid JSON array, no markdown:
                [{
                "building": 1,
                "location_description": "left foreground",
                "color": "white vinyl siding",
                "stories": 1,
                "structure_type": "residential wood building",
                "plan_type": "small rectangular plan",
                "roof_type": "gable roof",
                "roofCovering": {"obs": "brief observation", "ds": "DS0-DS4"},
                "windowDoor": {"obs": "brief observation", "ds": "DS0-DS4"},
                "roofSheathing": {"obs": "brief observation", "ds": "DS0-DS4"},
                "roofWall": {"obs": "brief observation", "ds": "DS0-DS4"},
                "overall": "DS0-DS4",
                "confidence_pct": 83,
                "rationale": "1 sentence referencing the rightmost column rule"
                }]"""

    messages = [{
        "role": "user",
        "content": [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": ext,
                    "data": b64
                }
            },
            {"type": "text", "text": prompt}
        ]
    }]

    if provider == "claude":
        import os
        key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        print(f"Received key: '{key}...' length={len(key)}")
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": key,
                    "anthropic-version": "2023-06-01",
                },
                json={"model": "claude-sonnet-4-20250514", "max_tokens": 1200, "messages": messages},
                timeout=60.0,
            )
            print(f"Anthropic status: {resp.status_code}")
            # print(f"Anthropic response: {resp.text}")
        return JSONResponse(content=resp.json(), status_code=resp.status_code)

    raise HTTPException(status_code=400, detail=f"Provider '{provider}' not yet supported.")

@app.post("/chat", tags=["AI Analysis"])
async def chat(
    payload: str = Form(...),
    api_key: str = Query(default=""),
    file: Optional[UploadFile] = File(default=None),
):
    import httpx, os, json, base64
    key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    
    data = json.loads(payload)
    
    # If an image file is provided, inject it into the last user message
    if file:
        raw = await file.read()
        b64 = base64.standard_b64encode(raw).decode("utf-8")
        media_type = file.content_type or "image/jpeg"
        
        # Find the last user message and add image to it
        messages = data.get("messages", [])
        for i in range(len(messages) - 1, -1, -1):
            if messages[i]["role"] == "user":
                # Convert text content to multimodal content
                existing_text = messages[i]["content"]
                if isinstance(existing_text, str):
                    messages[i]["content"] = [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": b64
                            }
                        },
                        {"type": "text", "text": existing_text}
                    ]
                break
        data["messages"] = messages

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "Content-Type": "application/json",
                "x-api-key": key,
                "anthropic-version": "2023-06-01",
            },
            json=data,
            timeout=60.0,
        )
    return JSONResponse(content=resp.json(), status_code=resp.status_code)