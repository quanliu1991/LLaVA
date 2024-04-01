from typing import List, Dict, Union, Optional,Text
from pydantic import BaseModel


class QueryBody(BaseModel):
    model_id: str
    image: str = None
    src_type: str = 'url'
    text: Union[str, List]
    choices: List[Text] = []
    temperature: float = 0.2
    max_tokens: int = 1024
    top_p: float = 1.0
    initial_prompt: str = None

class Prompt(BaseModel):
    user: str
    assistant: Optional[str]

class Records(BaseModel):
    image: str = None
    src_type: str = 'url'
    records: List[Prompt]
    choices: List[Text] = []


class BatchQueryBody(BaseModel):
    model_id: str
    prompts: List[Records]
    temperature: float = 0.2
    max_tokens: int = 1024
    top_p: float = 1.0
    initial_prompt: str = None    # used by banchmark
    input_tokens_number: int = None
    output_tokens_number: int = None
