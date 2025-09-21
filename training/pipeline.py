import argparse
import json
import os
import time
import uuid

from supabase import create_client

# 可選：from ultralytics import YOLO  或 subprocess 呼叫 `yolo` CLI

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
sb = create_client(SUPABASE_URL, SUPABASE_KEY)


def load_thresholds():
    r = (
        sb.table("system_flags")
        .select("value")
        .eq("key", "auto_train_threshold")
        .single()
        .execute()
        .data
    )
    return r["value"] if r else {"min_new_samples": 200, "min_label_ratio": 0.7}


def count_candidates():
    total = (
        sb.table("dataset_samples")
        .select("id", count="exact")
        .eq("is_labeled", True)
        .execute()
        .count
    )
    all_rows = sb.table("dataset_samples").select("id", count="exact").execute().count
    ratio = (total / all_rows) if all_rows else 0.0
    return total, ratio


def start_run(meta):
    return (
        sb.table("training_runs")
        .insert(
            {
                "status": "running",
                "triggered_by": meta.get("triggered_by", "auto"),
                "data_count": meta["data_count"],
                "params_json": meta["params"],
            }
        )
        .execute()
        .data[0]
    )


def finish_run(run_id, status, metrics=None, artifacts=None, err=None):
    sb.table("training_runs").update(
        {
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "status": status,
            "metrics_json": metrics or {},
            "artifacts": artifacts or {},
            "error_msg": err,
        }
    ).eq("id", run_id).execute()


def train_and_export(meta):
    # TODO: 匯出資料、訓練 YOLO、生成 best.pt 與 metrics.json
    # 這裡先放假資料骨架
    weights_path = f"/tmp/best_{uuid.uuid4().hex}.pt"
    metrics = {"f1": 0.84, "acc": 0.90}
    artifacts = {"weights": weights_path, "metrics": "/tmp/metrics.json"}
    return {"metrics": metrics, "artifacts": artifacts}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()

    if args.run_id:
        run = (
            sb.table("training_runs")
            .select("*")
            .eq("id", args.run_id)
            .single()
            .execute()
            .data
        )
        try:
            r = train_and_export({"params": run.get("params_json", {})})
            finish_run(
                run["id"],
                "pending_review",
                metrics=r["metrics"],
                artifacts=r["artifacts"],
            )
        except Exception as e:
            finish_run(run["id"], "failed", err=str(e))
        return

    th = load_thresholds()
    total, ratio = count_candidates()
    if total < th["min_new_samples"] or ratio < th["min_label_ratio"]:
        print("Threshold not met, skip.")
        return

    meta = {
        "triggered_by": "auto",
        "data_count": total,
        "params": {"seed": 42, "epochs": 80, "imgsz": 640, "optimizer": "adamw"},
    }
    run = start_run(meta)
    try:
        r = train_and_export(meta)
        finish_run(
            run["id"], "pending_review", metrics=r["metrics"], artifacts=r["artifacts"]
        )
    except Exception as e:
        finish_run(run["id"], "failed", err=str(e))


if __name__ == "__main__":
    main()
