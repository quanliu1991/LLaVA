import time

from fastapi import APIRouter, BackgroundTasks
from starlette.requests import Request

from api.schemas.payload import QueryBody, BatchQueryBody
from api.schemas.response import Response, BatchResponse
from api.services.detect import Engine
import traceback
import logging

router = APIRouter()


@router.post("/process", response_model=Response, name="Answer")
async def detect_urls(request: Request, body: QueryBody) -> Response:
    s_time = time.time()
    engine: Engine = request.app.state.detector
    try:
        res = engine.predict(
            model_id=body.model_id,
            image=body.image,
            src_type=body.src_type,
            text=body.text,
            choices=body.choices,
            initial_prompt=body.initial_prompt,
            temperature=body.temperature, 
            max_tokens=body.max_tokens, 
            top_p=body.top_p
        )
    except Exception as error:
        logging.error(traceback.format_exc())
        resp = Response(took=(time.time() - s_time) * 1000, code=500, error=error)
        return resp
    resp = Response(took=(time.time() - s_time) * 1000, code=200, answer=res)
    return resp

@router.post("/batch_infer", response_model=BatchResponse, name="batch_infer")
async def detect_urls(request: Request, body: BatchQueryBody) -> BatchResponse:
    s_time = time.time()
    engine: Engine = request.app.state.detector
    # try:
    res = engine.batch_predict(
        model_id=body.model_id,
        prompts=body.prompts,
        initial_prompt=body.initial_prompt,
        temperature=body.temperature,
        max_tokens=body.max_tokens,
        top_p=body.top_p
    )
    # except Exception as error:
    #     logging.error(traceback.format_exc())
    #     resp = Response(took=(time.time() - s_time) * 1000, code=500, error=error)
    #     return resp
    resp = BatchResponse(took=(time.time() - s_time) * 1000, code=200, answer=res)
    return resp
