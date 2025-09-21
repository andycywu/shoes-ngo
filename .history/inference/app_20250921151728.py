import io, os, json, time, base64
from typing import List, Tuple
from fastapi import FastAPI, UploadFile, Form, Header, HTTPException
from PIL import Image
from ultralytics import YOLO
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from supabase import create_client, Client
import qrcode, imagehash, numpy as np, cv2
from pydantic import BaseModel, Field, ValidationError

# --- env ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
STAGE1 = os.getenv("STAGE1_MODEL_PATH", "./inference/models/stage1_sneaker_cls.pt")
STAGE2 = os.getenv("STAGE2_MODEL_PATH", "./inference/models/stage2_defects_cls.pt")
VLM_ID = os.getenv("VLM_ID", "Qwen/Qwen2-VL-2B-Instruct")
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "please-change-me")

# --- clients & models ---
sb: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
yolo_s1, yolo_s2 = YOLO(STAGE1), YOLO(STAGE2)
tok = AutoTokenizer.from_pretrained(VLM_ID)
proc = AutoProcessor.from_pretrained(VLM_ID)
vlm = AutoModelForCausalLM.from_pretrained(VLM_ID, device_map="auto", torch_dtype="auto")

app = FastAPI(title="Shoes NGO API")

# --- schema for VLM ---
class Prices(BaseModel):
    _90: Tuple[int,int] = Field(..., alias="90")
    _70: Tuple[int,int] = Field(..., alias="70")
    _50: Tuple[int,int] = Field(..., alias="50")
class VLMOut(BaseModel):
    summary: str
    defects: List[str]
    suggestion: str
    title_zh: str
    title_en: str
    desc: str
    prices: Prices

ALLOWED_MIME = {"image/jpeg","image/png","image/webp"}
MAX_PIXELS = 4096*4096
MAX_BYTES = 8*1024*1024

def require_admin(token: str | None):
    if token != ADMIN_TOKEN:
        raise HTTPException(403, "admin only")

def yolo_cls(model, img: Image.Image):
    r = model.predict(img, imgsz=640, conf=0.25, verbose=False)
    probs = getattr(r[0], "probs", None)
    names = model.names
    scores = {names[i]: float(probs.data[i]) for i in range(len(names))} if probs else {}
    top = max(scores, key=scores.get) if scores else "unknown"
    return top, scores

def prompt_json(is_sneaker, defects, brand, model_name):
    d = ", ".join(defects) if defects else "none"
    return (
      "Output STRICT JSON with keys: summary, defects[], suggestion(donate|resale|recycle), "
      "title_zh, title_en, desc(<=150 chars), prices{90:[l,u],70:[l,u],50:[l,u]}.\n"
      f"Type: {'sneaker' if is_sneaker else 'non-sneaker'}\nBrand:{brand or 'unknown'}\n"
      f"Model:{model_name or 'unknown'}\nDetected defects:{d}\n"
      "Rules: hole/split-off -> recycle; flat -> donate; else resale. Only JSON."
    )

def parse_vlm_json(txt: str) -> dict:
    try:
        raw = json.loads(txt.strip())
        parsed = VLMOut.model_validate(raw)
        prices = {"90": list(parsed.prices._90), "70": list(parsed.prices._70), "50": list(parsed.prices._50)}
        d = parsed.model_dump(by_alias=True)
        d["prices"] = prices
        return d
    except Exception:
        return {"summary":"鞋況中等","defects":[],"suggestion":"donate",
                "title_zh":"運動鞋","title_en":"Sneakers","desc":"一般使用痕跡，清潔後可再用。",
                "prices":{"90":[1000,1500],"70":[700,1000],"50":[400,700]}}

def run_vlm(img: Image.Image, prompt: str) -> dict:
    inputs = proc(images=img, text=prompt, return_tensors="pt").to(vlm.device)
    out = vlm.generate(**inputs, max_new_tokens=256)
    txt = tok.decode(out[0], skip_special_tokens=True)
    return parse_vlm_json(txt)

def phash_hex(img: Image.Image) -> str:
    return str(imagehash.phash(img))
def laplacian_blur(img: Image.Image) -> float:
    arr = np.array(img.convert("L"))
    return float(cv2.Laplacian(arr, cv2.CV_64F).var())
def mark_candidate(s1c: float|None, s2c: float|None, defects: list, sugg: str) -> bool:
    if s1c and s1c < 0.7: return True
    if s2c and s2c < 0.6: return True
    if ("good" in defects) and (sugg in ["donate","recycle"]): return True
    return False
def qr_b64(payload: dict) -> str:
    buf = io.BytesIO()
    qrcode.make(json.dumps(payload, ensure_ascii=False)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

@app.get("/healthz")
def healthz():
    return {"ok": True, "models": {"stage1": bool(yolo_s1), "stage2": bool(yolo_s2), "vlm": bool(vlm)}}

@app.post("/analyze")
async def analyze(img: UploadFile, user_email: str = Form(None), brand: str = Form(None), model_name: str = Form(None)):
    if img.content_type not in ALLOWED_MIME:
        return {"error":"unsupported file type"}
    raw = await img.read()
    if len(raw) > MAX_BYTES:
        return {"error":"file too large"}
    im = Image.open(io.BytesIO(raw)).convert("RGB")
    if im.width*im.height > MAX_PIXELS:
        im.thumbnail((4096,4096))

    top1, s1 = yolo_cls(yolo_s1, im)
    is_sneaker = (top1 == "sneaker")
    defects, s2 = [], {}
    if is_sneaker:
        top2, s2 = yolo_cls(yolo_s2, im)
        if top2 not in ["good","unknown"]:
            defects = [top2]

    js = run_vlm(im, prompt_json(is_sneaker, defects, brand, model_name))
    suggestion = js.get("suggestion","resale")
    prices = js.get("prices", {})

    item = sb.table("items").insert({
        "user_email": user_email, "image_url": None,
        "is_sneaker": is_sneaker, "defects_json": s2 or s1,
        "suggestion": suggestion, "price_ranges_json": prices,
        "listing_title_zh": js.get("title_zh"), "listing_title_en": js.get("title_en"),
        "listing_desc": js.get("desc"), "vlm_summary": js.get("summary"), "status":"created"
    }).execute().data[0]

    phex = phash_hex(im); blur = laplacian_blur(im)
    s1c = max(s1.values()) if isinstance(s1, dict) and s1 else None
    s2c = max(s2.values()) if isinstance(s2, dict) and s2 else None
    cand = mark_candidate(s1c or 1.0, s2c or 1.0, defects, suggestion)
    sb.table("dataset_samples").insert({
      "item_id": item["id"],
      "image_path": None,
      "phash": phex,
      "blur_score": blur,
      "stage1_pred": top1, "stage1_conf": s1c,
      "stage2_pred": defects[0] if defects else None, "stage2_conf": s2c,
      "vlm_suggestion": suggestion,
      "candidate_for_training": cand
    }).execute()

    qr = None
    if suggestion in ["donate","recycle"]:
        payload = {"item_id": item["id"], "route": suggestion.upper(), "ts": int(time.time())}
        sb.table("logistics").insert({"item_id": item["id"], "route": suggestion.upper(), "qr_payload": payload}).execute()
        qr = qr_b64(payload)

    return {"is_sneaker": is_sneaker, "defects": defects, "vlm": js, "qr_b64": qr}

@app.post("/admin/start_cold_start")
async def start_cold_start(x_admin_token: str | None = Header(None)):
    require_admin(x_admin_token)
    cold = sb.table("system_flags").select("value").eq("key","cold_start_required").single().execute().data["value"]
    min_samples = cold.get("min_samples", 80)
    total = sb.table("dataset_samples").select("id", count="exact").eq("is_labeled", True).execute().count
    if total < min_samples:
        return {"ok": False, "msg": f"not enough labeled samples: {total}/{min_samples}"}
    run = sb.table("training_runs").insert({
        "status":"running","triggered_by":"manual:admin","data_count": total,
        "params_json":{"seed":42,"epochs":80,"imgsz":640,"optimizer":"adamw","notes":"cold start"},
        "gate_notes":"cold-start initiated by admin"
    }).execute().data[0]
    return {"ok": True, "run_id": run["id"]}

@app.get("/admin/runs")
async def list_runs(x_admin_token: str | None = Header(None)):
    require_admin(x_admin_token)
    rows = sb.table("training_runs").select("*").order("started_at", desc=True).limit(50).execute().data
    return rows

@app.post("/admin/approve_run")
async def approve_run(run_id: str, model_name: str, version: str | None = None,
                      x_admin_token: str | None = Header(None)):
    require_admin(x_admin_token)
    row = sb.table("training_runs").select("*").eq("id", run_id).single().execute().data
    if row["status"] != "pending_review":
        raise HTTPException(400, f"run not pending_review: {row['status']}")
    version = version or time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    sb.table("model_registry").insert({
        "model_name": model_name, "run_id": run_id, "version": version,
        "metrics_json": row.get("metrics_json", {}), "artifacts": row.get("artifacts", {}),
        "approved_by": "admin@your.org"
    }).execute()
    sb.table("training_runs").update({
        "status": "succeeded", "approved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "approved_by": "admin@your.org"
    }).eq("id", run_id).execute()
    sb.table("system_flags").update({"value": {"enabled": False}}).eq("key","cold_start_required").execute()
    return {"ok": True, "message": f"run {run_id} approved and registered for {model_name}"}