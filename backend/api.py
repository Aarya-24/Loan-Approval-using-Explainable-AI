from pathlib import Path
from typing import Literal

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .predict import predict_applicant

ROOT_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT_DIR / "frontend"

app = FastAPI(
    title="ClearPath Loan Check",
    description="Loan approval probability estimate powered by the project model.",
    version="1.0.0",
)
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


class ApplicantInput(BaseModel):
    age: int = Field(ge=18, le=100)
    annual_income: float = Field(gt=0)
    credit_score: int = Field(ge=300, le=850)
    employment_status: Literal["Employed", "Self-Employed", "Unemployed"]
    loan_amount: float = Field(gt=0)
    collateral: float = Field(ge=0)
    monthly_debt: float = Field(ge=0)


@app.get("/")
def home():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/predict")
def predict(payload: ApplicantInput):
    result = predict_applicant(
        age=payload.age,
        income=payload.annual_income,
        credit_score=payload.credit_score,
        employment_status=payload.employment_status,
        loan_amount=payload.loan_amount,
        collateral=payload.collateral,
        monthly_debt=payload.monthly_debt,
    )
    return result
