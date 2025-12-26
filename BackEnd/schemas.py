from pydantic import BaseModel
from datetime import datetime

class PredictRequest(BaseModel):
    input_data: str

class PredictResponse(BaseModel):
    id: int
    result: str
    confidence: float
    created_at: datetime

    class Config:
        from_attributes = True # SQLAlchemy 객체를 Pydantic으로 자동 변환