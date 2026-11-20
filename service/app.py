import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request

from model.conditional_curve import DEFAULT_MODEL_PATH
from service.conditional_forecasts import ConditionalForecasts

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app):
    model_path = Path(os.environ.get('MODEL_PATH') or DEFAULT_MODEL_PATH)
    if model_path.suffix == '.npz':
        forecasts = ConditionalForecasts(model_path)
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        from service.predictor import Predictor
        from service.updater import DataUpdater
        forecasts = DataUpdater(min_event_id=306, update_interval=1800)
        forecasts.set_predictor(Predictor(str(model_path)))
    app.state.forecasts = forecasts
    logger.info('Loaded prediction model %s', model_path)
    forecasts.start()
    try:
        yield
    finally:
        forecasts.stop()


app = FastAPI(title='Event Prediction Service', lifespan=lifespan)


@app.get('/predictions/{event_id}')
def get_predictions(request: Request, event_id: int, server: int = Query(3), tier: int = Query(1000)):
    payload = request.app.state.forecasts.get_predictions(event_id, server, tier)
    if payload is None:
        raise HTTPException(status_code=404, detail='No data for this event/server/tier')
    return payload


@app.get('/health')
def health(request: Request):
    result = {'status': 'ok'}
    forecasts = request.app.state.forecasts
    if isinstance(forecasts, ConditionalForecasts):
        result.update(model=forecasts.predictor.model.metadata['version'], model_sha256=forecasts.predictor.model_hash)
    return result
