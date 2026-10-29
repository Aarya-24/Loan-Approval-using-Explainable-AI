from pathlib import Path
from typing import Literal
import secrets
import string
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .predict import predict_applicant
from .persistence import find_loan_request, save_loan_request

ROOT_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT_DIR / "frontend"

app = FastAPI(
    title="ClearPath Loan Check",
    description="Loan approval probability estimate powered by the project model.",
    version="1.0.0",
)
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


class ApplicantInput(BaseModel):
    name: str = Field(min_length=1, max_length=120)
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
    alphabet = string.ascii_uppercase + string.digits
    request_id = "".join(secrets.choice(alphabet) for _ in range(8))
    record = {
        "loan_request_id": request_id,
        "name": payload.name.strip(),
        "age": payload.age,
        "annual_income": payload.annual_income,
        "credit_score": payload.credit_score,
        "employment_status": payload.employment_status,
        "loan_amount": payload.loan_amount,
        "collateral": payload.collateral,
        "monthly_debt": payload.monthly_debt,
        "probability": result["probability"],
        "decision": result["decision"],
        "created_at": datetime.now(timezone.utc).replace(tzinfo=None),
    }
    try:
        save_loan_request(record)
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Could not save the loan request. Check database configuration and availability.") from exc
    return {**result, "loan_request_id": request_id}


@app.get("/api/loan-requests/{loan_request_id}")
def get_loan_request(loan_request_id: str):
    try:
        record = find_loan_request(loan_request_id.strip().upper())
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Loan database is unavailable or not configured.") from exc
    if record is None:
        raise HTTPException(status_code=404, detail="Loan request ID does not exist.")
    return record
