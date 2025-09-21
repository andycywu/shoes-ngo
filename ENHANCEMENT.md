# pHash/Blur 分數與 Dataset Samples 擴展

## 在 app.py 中加入以下內容：

```python
import cv2
import numpy as np
import imagehash

# 在 inference/app.py 的 imports 區塊加入：
# import cv2
# import numpy as np
# import imagehash

# 在 inference/requirements.txt 加入：
# opencv-python
# imagehash

# 新增函數
def calculate_blur_score(img: Image.Image):
    """計算圖片模糊分數 (Laplacian variance)"""
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def calculate_phash(img: Image.Image):
    """計算感知哈希"""
    return str(imagehash.phash(img))

def should_mark_for_training(is_sneaker, defects, blur_score, confidence_scores):
    """決定是否標記為訓練候選"""
    # 低信心樣本
    max_confidence = max(confidence_scores.values()) if confidence_scores else 0
    
    # 條件：模糊、低信心、或邊緣案例
    return (
        blur_score < 100 or  # 模糊圖片
        max_confidence < 0.7 or  # 低信心預測
        (is_sneaker and not defects)  # 可能的假陰性
    )

# 在 /analyze 端點中加入這些計算
def enhanced_analyze():
    # ... 現有的 YOLO 和 VLM 處理 ...
    
    # 新增計算
    blur_score = calculate_blur_score(im)
    phash = calculate_phash(im)
    
    # 判斷是否需要加入訓練資料集
    candidate_for_training = should_mark_for_training(
        is_sneaker, defects, blur_score, s1
    )
    
    # 寫入 dataset_samples 表
    sb.table("dataset_samples").insert({
        "item_id": item["id"],
        "phash": phash,
        "blur_score": blur_score,
        "confidence_scores": s1,
        "candidate_for_training": candidate_for_training,
        "annotation_status": "pending" if candidate_for_training else "auto_labeled"
    }).execute()
```

## Supabase SQL 擴展：

```sql
-- 加入到 infra/supabase.sql

create table if not exists dataset_samples (
  id uuid primary key default gen_random_uuid(),
  item_id uuid references items(id) on delete cascade,
  phash text,
  blur_score float,
  confidence_scores jsonb,
  candidate_for_training boolean default false,
  annotation_status text default 'pending' check (annotation_status in ('pending','annotated','auto_labeled')),
  manual_labels jsonb,
  created_at timestamptz default now()
);

-- 建立索引
create index if not exists idx_dataset_samples_phash on dataset_samples(phash);
create index if not exists idx_dataset_samples_candidate on dataset_samples(candidate_for_training);
```

## 使用建議：

1. **資料收集**：每次分析都會自動記錄 pHash 和模糊分數
2. **去重**：透過 pHash 可以發現重複上傳的圖片
3. **品質篩選**：模糊或低信心的樣本會被標記需要人工標註
4. **持續學習**：累積足夠的候選樣本後，可以重新訓練模型

這樣的設計讓您的系統從第一天就開始收集有價值的訓練資料！