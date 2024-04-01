from pydantic import BaseModel
from typing import List, Optional

class Answer(BaseModel):
    content:str
    input_tokens:int
    output_tokens:int
    probs: dict = {}
    mean_prob: float = None


class Response(BaseModel):
    code: int
    took: int
    answer: Optional[Answer]
    error: Optional[str]

class BatchResponse(BaseModel):
    code: int
    took: int
    answer: Optional[List[Answer]]
    error: Optional[str]