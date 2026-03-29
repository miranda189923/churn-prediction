from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
from io import StringIO
from .models import ChurnRequest
from .utils import predict_single, predict_batch

app = FastAPI(title="Telco Churn Predictor API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
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
        
        # Return rich JSON instead of forcing a download
        return JSONResponse(content={
            "success": True,
            "message": f"Successfully processed {len(result_df)} customers",
            "summary": {
                "total_customers": len(result_df),
                "predicted_churn_count": int((result_df["prediction"] == "Churn").sum()),
                "churn_rate_percent": round((result_df["prediction"] == "Churn").mean() * 100, 1),
                "high_risk_count": int((result_df["risk_level"] == "High").sum()),
                "medium_risk_count": int((result_df["risk_level"] == "Medium").sum()),
                "low_risk_count": int((result_df["risk_level"] == "Low").sum())
            },
            "data": result_df.to_dict(orient="records"),
            "columns": list(result_df.columns)
        })
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": True}