# Shoes NGO (Ultra Low Cost)

## Setup
1) `cp .env.example .env` 並填入 Supabase 參數
2) 放入 YOLO 權重：`inference/models/stage1_sneaker_cls.pt`, `stage2_defects_cls.pt`

## Run (local)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r inference/requirements.txt
export $(cat .env | xargs)
uvicorn inference.app:app --host 0.0.0.0 --port 7860
```

## Colab
- 在 Colab `!git clone https://github.com/<you>/shoes-ngo.git`
- `pip install -r shoes-ngo/inference/requirements.txt`
- `os.environ[...]` 設好環境變數或寫 `.env` parser
- 路徑指向 `/content/shoes-ngo/inference/models/*.pt`
- 啟 uvicorn + 用 cloudflared 暴露網址

## Deploy
- 最省：Colab + cloudflared（免費）
- 或 Railway/Render 免費層（設環境變數，入口 `uvicorn inference.app:app --host 0.0.0.0 --port 7860`）

## Frontend
- `frontend-min/index.html` 可直接丟 Vercel 靜態託管。