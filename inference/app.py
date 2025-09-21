import io, os, json, time, base64
from fastapi import FastAPI, UploadFile, Form
from PIL import Image
from ultralytics import YOLO
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from supabase import create_client, Client
import qrcode

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
STAGE1 = os.getenv("STAGE1_MODEL_PATH", "./inference/models/stage1_sneaker_cls.pt")
STAGE2 = os.getenv("STAGE2_MODEL_PATH", "./inference/models/stage2_defects_cls.pt")
VLM_ID = os.getenv("VLM_ID", "Qwen/Qwen2-VL-2B-Instruct")

sb: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
yolo_s1, yolo_s2 = YOLO(STAGE1), YOLO(STAGE2)

tok = AutoTokenizer.from_pretrained(VLM_ID)
proc = AutoProcessor.from_pretrained(VLM_ID)
vlm = AutoModelForCausalLM.from_pretrained(VLM_ID, device_map="auto", torch_dtype="auto")

app = FastAPI(title="Shoes NGO API")

def yolo_cls(model, img):
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

def run_vlm(img: Image.Image, prompt: str):
    inputs = proc(images=img, text=prompt, return_tensors="pt").to(vlm.device)
    out = vlm.generate(**inputs, max_new_tokens=256)
    txt = tok.decode(out[0], skip_special_tokens=True).strip()
    try:
        return json.loads(txt)
    except:
        return {"summary":"鞋況中等","defects":[],"suggestion":"donate",
                "title_zh":"運動鞋","title_en":"Sneakers","desc":"一般使用痕跡，清潔後可再用。",
                "prices":{"90":[1000,1500],"70":[700,1000],"50":[400,700]}}

def make_qr_payload(item_id, route):
    return {"item_id": item_id, "route": route, "ts": int(time.time())}

def qr_b64(payload):
    buf = io.BytesIO()
    qrcode.make(json.dumps(payload, ensure_ascii=False)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

@app.post("/analyze")
async def analyze(img: UploadFile, user_email: str = Form(None), brand: str = Form(None), model_name: str = Form(None)):
    im = Image.open(io.BytesIO(await img.read())).convert("RGB")
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

    qr = None
    if suggestion in ["donate","recycle"]:
        route = suggestion.upper()
        payload = make_qr_payload(item["id"], route)
        sb.table("logistics").insert({"item_id": item["id"], "route": route, "qr_payload": payload}).execute()
        qr = qr_b64(payload)

    return {"is_sneaker": is_sneaker, "defects": defects, "vlm": js, "qr_b64": qr}