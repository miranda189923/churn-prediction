from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import pandas as pd
from io import StringIO
from .models import ChurnRequest
from .utils import predict_single, predict_batch

app = FastAPI(title="Telco Churn Predictor API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # change to your frontend URL in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def single_predict(request: ChurnRequest):
    try:
        return predict_single(request.model_dump())
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/predict_batch")
async def batch_predict(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Only CSV files allowed")
    try:
        content = await file.read()
        df = pd.read_csv(StringIO(content.decode()))
        result_df = predict_batch(df)
        
        output = StringIO()
        result_df.to_csv(output, index=False)
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=churn_predictions.csv"}
        )
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": True}