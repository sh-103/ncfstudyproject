from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from fastapi.middleware.cors import CORSMiddleware

import models, schemas
from database import engine, get_db

# 서버 시작 시 테이블 생성 (현실적으로는 Alembic 추천)
models.Base.metadata.create_all(bind=engine)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict", response_model=schemas.PredictResponse)
async def predict_and_store(request: schemas.PredictRequest, db: Session = Depends(get_db)):
    try:
        # 1. 추론 수행 (가상 로직)
        inference_result = f"Result for {request.input_data}"
        confidence_score = 0.98

        # 2. DB 객체 생성
        new_log = models.InferenceLog(
            input_text=request.input_data,
            prediction=inference_result,
            confidence=confidence_score
        )
        
        # 3. DB 저장 및 확정
        db.add(new_log)
        db.commit()
        db.refresh(new_log)

        return new_log
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database Error: {str(e)}")