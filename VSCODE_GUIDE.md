# 📁 VS Code 工作區使用指南

## 🚀 快速開始

### 1. 開啟工作區
```bash
code shoes-ngo.code-workspace
```

### 2. 自動設置環境
- 按 `Ctrl/Cmd + Shift + P`
- 輸入 "Tasks: Run Task"
- 選擇 "Setup Complete Project"

### 3. 啟動開發服務器
- 按 `F5` 啟動調試模式
- 或使用任務 "Start FastAPI Server"

## 🛠️ 工作區功能

### 📋 內建任務 (Ctrl/Cmd + Shift + P → Tasks: Run Task)
- **Setup Virtual Environment**: 建立虛擬環境
- **Install Dependencies**: 安裝依賴套件
- **Start FastAPI Server**: 啟動 API 服務器
- **Run Tests**: 執行單元測試
- **Format Code**: 格式化程式碼
- **Setup Complete Project**: 完整專案設置

### 🐛 調試配置 (F5 或 Run and Debug 面板)
- **FastAPI Development**: 啟動 FastAPI 開發服務器
- **Python: Current File**: 執行當前 Python 檔案
- **Model Training Script**: 執行模型訓練腳本

### 🧪 測試
- **API 測試**: 使用 `api_tests.http` 檔案測試 API
- **單元測試**: `tests/` 目錄中的 pytest 測試

### 📦 推薦擴展
工作區會自動推薦安裝以下擴展：
- Python, Pylint, Black Formatter
- Jupyter Notebooks
- GitHub Copilot & Copilot Chat
- REST Client (用於 API 測試)
- Thunder Client (API 測試客戶端)

## 📂 目錄結構
```
shoes-ngo/
├── .vscode/              # VS Code 配置
│   ├── settings.json     # 工作區設定
│   ├── launch.json       # 調試配置
│   ├── tasks.json        # 任務配置
│   └── extensions.json   # 推薦擴展
├── scripts/              # 工具腳本
│   └── train_models.py   # 模型訓練腳本
├── tests/                # 測試檔案
│   └── test_api.py       # API 測試
├── test_images/          # 測試圖片
├── inference/            # 推理服務
│   ├── app.py           # FastAPI 應用
│   ├── requirements.txt  # 生產依賴
│   └── models/          # 模型檔案目錄
├── frontend-min/         # 最小前端
├── infra/               # 基礎設施
├── api_tests.http       # REST Client 測試
├── requirements-dev.txt  # 開發依賴
└── shoes-ngo.code-workspace # 工作區檔案
```

## 🔧 環境設定

### Python 環境
- 自動使用 `.venv/bin/python`
- 自動載入 `.env` 檔案
- 啟用格式化和 linting

### 快速命令
```bash
# 安裝開發依賴
.venv/bin/pip install -r requirements-dev.txt

# 格式化程式碼
.venv/bin/black inference/ scripts/

# 執行測試
.venv/bin/pytest tests/ -v

# 啟動服務器
.venv/bin/uvicorn inference.app:app --host 0.0.0.0 --port 7860 --reload
```

## 📱 前端測試
- 開啟 `frontend-min/index.html`
- 使用 Live Server 擴展即時預覽
- 設定 API base URL 為 `http://localhost:7860`

## 🤖 Copilot 整合
- 使用 `Ctrl/Cmd + I` 開啟 Copilot Chat
- 在任何檔案中使用 Copilot 自動完成
- 詢問專案相關問題和程式碼建議

## 🚀 部署
- **本機開發**: 使用 VS Code 調試功能
- **Colab 訓練**: 執行 `scripts/train_models.py`
- **生產部署**: 使用 `inference/requirements.txt`