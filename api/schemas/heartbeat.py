from pydantic import BaseModel


class Heartbeat(BaseModel):
    is_alive: bool
