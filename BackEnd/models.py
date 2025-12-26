from sqlalchemy import Column, Integer, String, Float, DateTime
from datetime import datetime
from database import Base

class InferenceLog(Base):
    __tablename__ = "inference_logs"

    id = Column(Integer, primary_key=True, index=True)
    input_text = Column(String(500))
    prediction = Column(String(100))
    confidence = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)