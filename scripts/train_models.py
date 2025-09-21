"""
從 Roboflow 匯入資料並訓練兩階段鞋類分析模型
在 VS Code 中可以直接執行或調試此腳本
"""

import os
import shutil
from pathlib import Path
from roboflow import Roboflow
from ultralytics import YOLO

# 專案根目錄
ROOT_DIR = Path(__file__).parent.parent
MODELS_DIR = ROOT_DIR / "inference" / "models"

def setup_roboflow(api_key: str = None):
    """設定 Roboflow 連接"""
    if api_key:
        rf = Roboflow(api_key=api_key)
    else:
        rf = Roboflow()  # 公開專案不需要 API key
    return rf

def train_stage1(rf, workspace: str, project_name: str, version: int = 1):
    """訓練 Stage 1: Sneaker vs Non-sneaker"""
    print("🚀 開始訓練 Stage 1: Sneaker/Non-sneaker 分類...")
    
    # 下載資料集
    project = rf.workspace(workspace).project(project_name)
    dataset = project.version(version).download("yolov8")
    
    # 載入預訓練模型
    model = YOLO('yolov8n-cls.pt')
    
    # 訓練參數
    train_args = {
        'data': dataset.location,
        'epochs': 100,
        'imgsz': 640,
        'batch': 16,
        'name': 'stage1_sneaker_cls',
        'patience': 10,
        'save': True,
        'plots': True,
        'val': True,
        'project': str(ROOT_DIR / 'runs' / 'classify')
    }
    
    # 訓練
    results = model.train(**train_args)
    
    # 驗證
    metrics = model.val()
    print(f"✅ Stage 1 訓練完成！準確率: {metrics.top1:.3f}")
    
    return model, results, metrics

def train_stage2(rf, workspace: str, project_name: str, version: int = 1):
    """訓練 Stage 2: 瑕疵檢測"""
    print("🚀 開始訓練 Stage 2: 瑕疵檢測分類...")
    
    # 下載資料集
    project = rf.workspace(workspace).project(project_name)
    dataset = project.version(version).download("yolov8")
    
    # 載入預訓練模型
    model = YOLO('yolov8n-cls.pt')
    
    # 訓練參數
    train_args = {
        'data': dataset.location,
        'epochs': 100,
        'imgsz': 640,
        'batch': 16,
        'name': 'stage2_defects_cls',
        'patience': 10,
        'save': True,
        'plots': True,
        'val': True,
        'project': str(ROOT_DIR / 'runs' / 'classify')
    }
    
    # 訓練
    results = model.train(**train_args)
    
    # 驗證
    metrics = model.val()
    print(f"✅ Stage 2 訓練完成！準確率: {metrics.top1:.3f}")
    
    return model, results, metrics

def save_models():
    """儲存訓練好的模型到 inference/models/"""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    runs_dir = ROOT_DIR / 'runs' / 'classify'
    
    # 複製 Stage 1 模型
    stage1_path = runs_dir / 'stage1_sneaker_cls' / 'weights' / 'best.pt'
    if stage1_path.exists():
        shutil.copy(stage1_path, MODELS_DIR / 'stage1_sneaker_cls.pt')
        print(f"✅ Stage 1 模型已儲存至 {MODELS_DIR}/stage1_sneaker_cls.pt")
    else:
        print(f"⚠️  Stage 1 模型檔案不存在: {stage1_path}")
    
    # 複製 Stage 2 模型
    stage2_path = runs_dir / 'stage2_defects_cls' / 'weights' / 'best.pt'
    if stage2_path.exists():
        shutil.copy(stage2_path, MODELS_DIR / 'stage2_defects_cls.pt')
        print(f"✅ Stage 2 模型已儲存至 {MODELS_DIR}/stage2_defects_cls.pt")
    else:
        print(f"⚠️  Stage 2 模型檔案不存在: {stage2_path}")

def test_models():
    """測試訓練好的模型"""
    print("🧪 測試模型...")
    
    stage1_model_path = MODELS_DIR / 'stage1_sneaker_cls.pt'
    stage2_model_path = MODELS_DIR / 'stage2_defects_cls.pt'
    
    if stage1_model_path.exists():
        stage1_model = YOLO(str(stage1_model_path))
        print("📊 Stage 1 類別:", stage1_model.names)
    else:
        print("⚠️  Stage 1 模型檔案不存在")
        return None, None
    
    if stage2_model_path.exists():
        stage2_model = YOLO(str(stage2_model_path))
        print("📊 Stage 2 類別:", stage2_model.names)
    else:
        print("⚠️  Stage 2 模型檔案不存在")
        return stage1_model, None
    
    return stage1_model, stage2_model

def main():
    """主函數"""
    # 配置 - 請修改為您的實際專案資訊
    CONFIG = {
        "api_key": None,  # 如果是私有專案，請填入您的 API key
        "workspace": "YOUR_WORKSPACE_NAME",
        "stage1_project": "sneaker-classification",
        "stage2_project": "shoe-defects",
        "version": 1
    }
    
    print("🎯 開始兩階段模型訓練...")
    print(f"📁 專案根目錄: {ROOT_DIR}")
    print(f"📁 模型儲存目錄: {MODELS_DIR}")
    
    # 設定 Roboflow
    try:
        rf = setup_roboflow(CONFIG["api_key"])
        print("✅ Roboflow 連接成功")
    except Exception as e:
        print(f"❌ Roboflow 連接失敗: {e}")
        return
    
    # 訓練 Stage 1
    try:
        model_stage1, results_stage1, metrics_stage1 = train_stage1(
            rf, CONFIG["workspace"], CONFIG["stage1_project"], CONFIG["version"]
        )
    except Exception as e:
        print(f"❌ Stage 1 訓練失敗: {e}")
        return
    
    # 訓練 Stage 2
    try:
        model_stage2, results_stage2, metrics_stage2 = train_stage2(
            rf, CONFIG["workspace"], CONFIG["stage2_project"], CONFIG["version"]
        )
    except Exception as e:
        print(f"❌ Stage 2 訓練失敗: {e}")
        return
    
    # 儲存模型
    save_models()
    
    # 測試模型
    test_stage1, test_stage2 = test_models()
    
    print("🎉 訓練完成！")
    print("💡 現在您可以啟動 FastAPI 服務來測試完整系統")
    print("🚀 使用 VS Code 任務 'Start FastAPI Server' 或按 F5 啟動調試")

if __name__ == "__main__":
    main()