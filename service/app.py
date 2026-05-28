# service/app.py
import logging
import sys
import os
from fastapi import FastAPI, Query, HTTPException
from .predictor import Predictor
from .updater import DataUpdater
from data.constants import t_min

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Event Prediction Service")
predictor: Predictor = None
updater: DataUpdater = None

@app.on_event("startup")
async def startup():
    global predictor, updater
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    model_path = os.environ.get("MODEL_PATH", "saved_models/moe.keras")

    logger.info("Loading model...")
    predictor = Predictor(model_path)

    updater = DataUpdater(
        min_event_id=306,
        update_interval=1800,  # 30分钟更新一次
        t_min=t_min,   # 可以从 Config 读取，但为简洁直接给出
        batch_size=256,
    )
    updater.set_predictor(predictor)
    updater.start()

@app.get("/predictions/{event_id}")
async def get_predictions(
    event_id: int,
    server: int = Query(3, description="服务器ID"),
    tier: int = Query(1000, description="档位值"),
):

    # 获取最新切片（内部自动判断是否活跃并更新缓存）
    latest_slice = updater.ensure_latest_slice(event_id, server, tier)
    # 如果活动未在进行或切点太早，latest_slice 可能为 None，这是预期行为
    # 获取历史切点列表
    history = updater.cache.get_history(event_id, server, tier)
    if history is None:
        raise HTTPException(status_code=404, detail="No data for this event/server/tier")

    return {
        "event_id": event_id,
        "server": server,
        "tier": tier,
        "history": history,
        "latest_slice": latest_slice,
    }

@app.get("/health")
def health():
    return {"status": "ok"}
