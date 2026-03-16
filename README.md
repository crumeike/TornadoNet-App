# TornadoNet-App

**Automated street-level building damage detection for post-tornado disaster assessment.**

TornadoNet-App is an AI-powered assessment tool combining fine-tuned object detection models with LLM-based structural analysis. It classifies building damage according to the [IN-CORE](https://incore.ncsa.illinois.edu/) DS0–DS4 framework for wood-frame residential archetypes (T1–T5), producing structured outputs suitable for NIST/FEMA reporting workflows.

Built under CRADA CN-24-0590 in collaboration with Johns Hopkins University, the University of Alabama, the University of South Alabama, and NIST.

---

## Features

- **Three TornadoNet detection models**: RT-DETR-L with ordinal supervision (best severity grading), YOLO11x (best mAP), and YOLO11n (fastest, field-deployable)
- **IN-CORE DS0–DS4 classification**: five-level damage states for wood-frame buildings (archetypes T1–T5)
- **Bounding box detection preview**: color-coded boxes drawn on uploaded images; click any detection row to highlight its building
- **AI structural assessment**: Claude, GPT-4o, Gemini, or Grok analyze each detected building using the T1–T5 decision matrix (roof covering, window/door, roof sheathing, roof-to-wall connection) and return a final DS assignment per element
- **NIST-style report generation**: auto-generated damage narrative and response recommendations exportable as PDF
- **AI triage assistant**: conversational assistant grounded in session detections and the IN-CORE framework
- **Video frame extraction**: select keyframes from dashcam or 360° video for single-frame inference
- **Multi-provider LLM support**: Anthropic, OpenAI, Google, xAI

---

## Models

All checkpoints are available on HuggingFace: [`crumeike/tornadonet-checkpoints`](https://huggingface.co/crumeike/tornadonet-checkpoints)

| Model | mAP@0.5 | Ordinal Acc | MAOE | FPS | Download* | Notes |
|---|---|---|---|---|---|
| RT-DETR-L + Ordinal (ψ=0.5, K=1) | 44.70% | 91.15% | 0.56 | 78 | [📥](https://huggingface.co/crumeike/tornadonet-checkpoints/tree/main/tornadonet-rtdetr-l-ordinal-psi0.5-k1) | Best severity grading |
| YOLO11x Baseline | 46.05% | 85.20% | 0.76 | 66 | [📥](https://huggingface.co/crumeike/tornadonet-checkpoints/tree/main/tornadonet-yolo11-x-baseline) | Best detection mAP |
| YOLO11n Baseline | 41.14% | 84.79% | 0.77 | 239 | [📥](https://huggingface.co/crumeike/tornadonet-checkpoints/tree/main/tornadonet-yolo11-n-baseline) | Fastest, field-deployable |
\* *Values represent mean ± std across 3 random seeds (paper). Downloaded checkpoints are from the best-performing seed.*
Evaluated on 3,333 street-view images (8,890 building instances) collected after the December 2021 Midwest tornado outbreak.

---

## Repository Structure

```
TornadoNet/
├── api/
│   └── main.py              # FastAPI backend — loads .pt checkpoints, exposes /predict
├── checkpoints/             # Place .pt files here (gitignored, download from HuggingFace)
│   └── .gitkeep
├── frontend/
│   └── index.html           # Standalone HTML demo (zero setup, paste API key and go)
├── requirements.txt
├── .env.example             # Copy to .env and fill in API keys
├── .gitignore
└── README.md
```

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/crumeike/TornadoNet-App
cd TornadoNet-App
pip install -r requirements.txt
```

### 2. Download checkpoints

Download the relevant `.pt` files from [HuggingFace](https://huggingface.co/crumeike/tornadonet-checkpoints) and place them in `checkpoints/`:

```
checkpoints/
├── tornadonet-rtdetr-l-ordinal-psi0.5-k1.pt
├── tornadonet-yolo11-x-baseline.pt
└── tornadonet-yolo11-n-baseline.pt
```

Or set `HF_TOKEN` in your `.env` and the API will download them automatically on first request.


### 3. Configure environment

```bash
cp .env.example .env
# Edit .env and add your API keys
```

### 4. Start the backend

```bash
uvicorn api.main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`. Visit `http://localhost:8000/docs` for the interactive Swagger UI.

### 5. Open the frontend

Open `frontend/index.html` in your browser, paste your Anthropic API key in Settings, set the TornadoNet URL to `http://localhost:8000`, and click Test connection.

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Connection test, loaded model list |
| `GET` | `/models` | Model metadata and statistics |
| `POST` | `/predict` | Single image inference |
| `POST` | `/predict/batch` | Batch image inference (up to 50) |
| `POST` | `/predict/video-frame` | Extract a video frame and run inference |

### Example: single image inference

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@building.jpg" \
  -F "model_id=rtdetr-ord" \
  -F "conf=0.25"
```

### Response structure

```json
{
  "detections": [
    {
      "class_idx": 3,
      "ds_id": "DS3",
      "ds_name": "Extensive",
      "ds_color": "#d44e2a",
      "confidence": 0.8841,
      "bbox_xyxy": [120.5, 48.2, 380.1, 290.7],
      "bbox_xywh": [120.5, 48.2, 259.6, 242.5]
    }
  ],
  "detection_count": 1,
  "damage_distribution": {"DS0": 0, "DS1": 0, "DS2": 0, "DS3": 1, "DS4": 0},
  "annotated_image_b64": "<base64-encoded JPEG>",
  "inference_ms": 38.4,
  "model_id": "rtdetr-ord",
  "model_label": "RT-DETR-L + Ordinal Supervision (psi=0.5, K=1)"
}
```

---

## IN-CORE Damage States — T1–T5 Wood Frame Buildings

| State | Description | Roof Covering | Windows/Doors | Roof Sheathing | Roof-to-Wall |
|---|---|---|---|---|---|
| DS0 | Undamaged | No damage | No failures | No failure | Intact |
| DS1 | Slight | 2–15% damaged | 1 failure | No failure | Intact |
| DS2 | Moderate | 15–50% damaged | 2–3 failures | 1–3 sections | Intact |
| DS3 | Extensive | >50% damaged | >3 failures | >3 sections, <35% area | Intact |
| DS4 | Complete | >50% (typically) | >3 (typically) | >35% area | Failed |

---

## Citation

If you use TornadoNet in your research, please cite:

```bibtex
@misc{tornadonet2025,
  title   = {TornadoNet: Real-Time Building Damage Detection with Ordinal Supervision},
  year    = {2025},
  note    = {CRADA CN-24-0590. Johns Hopkins University, University of Alabama,
             University of South Alabama, NIST.
             \url{https://github.com/crumeike/TornadoNet}}
}
```

---

## Acknowledgments

This work was supported by the Center for Risk-Based Community Resilience Planning, a NIST-funded Center of Excellence (Cooperative Agreement 70NANB15H044), and SciServer (NSF Award ACI-1261715) at Johns Hopkins University.

---

## License

See ['LICENSE'](LICENSE) for terms. Model checkpoints and dataset are released for research use.
