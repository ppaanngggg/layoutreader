import os
import time
from typing import List

import torch
import uvicorn
from fastapi import FastAPI
from loguru import logger
from pydantic import BaseModel, Field
from transformers import LayoutLMv3ForTokenClassification

from v3.helpers import MAX_LEN, parse_logits, prepare_inputs, boxes2inputs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = (
    LayoutLMv3ForTokenClassification.from_pretrained(
        os.getenv("LOAD_PATH", "hantian/layoutreader")
    )
    .bfloat16()
    .to(device)
    .eval()
)
app = FastAPI()


@app.get("/config")
def get_config():
    return {
        "max_len": MAX_LEN,
    }


class PredictRequest(BaseModel):
    boxes: List[List[float]] = Field(
        ...,
        examples=[[[2, 2, 3, 3], [1, 1, 2, 2]]],
        description="Boxes of [left, top, right, bottom]",
    )
    width: float = Field(..., examples=[5], description="Page width")
    height: float = Field(..., examples=[5], description="Page height")


class PredictResponse(BaseModel):
    orders: List[int] = Field(..., examples=[[1, 0]], description="The order of spans.")
    elapsed: float = Field(
        ..., examples=[0.123], description="Elapsed time in seconds."
    )


def do_predict(boxes: List[List[int]]) -> List[int]:
    inputs = boxes2inputs(boxes)
    inputs = prepare_inputs(inputs, model)
    logits = model(**inputs).logits.cpu().squeeze(0)
    return parse_logits(logits, len(boxes))


@app.post("/predict")
def predict(request: PredictRequest) -> PredictResponse:
    x_scale = 1000.0 / request.width
    y_scale = 1000.0 / request.height

    boxes = []
    logger.info(f"Scale: {x_scale}, {y_scale}, Boxes len: {len(request.boxes)}")
    for left, top, right, bottom in request.boxes:
        left = round(left * x_scale)
        top = round(top * y_scale)
        right = round(right * x_scale)
        bottom = round(bottom * y_scale)
        assert (
            1000 >= right >= left >= 0 and 1000 >= bottom >= top >= 0
        ), f"Invalid box. right: {right}, left: {left}, bottom: {bottom}, top: {top}"
        boxes.append([left, top, right, bottom])

    start = time.time()
    orders = do_predict(boxes)
    ret = PredictResponse(orders=orders, elapsed=time.time() - start)
    logger.info(f"Input Length: {len(boxes)}, Predicted in {ret.elapsed:.3f}s.")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return ret


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
