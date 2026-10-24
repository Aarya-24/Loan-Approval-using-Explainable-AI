# "Age","AnnualIncome","CreditScore","EmploymentStatus","LoanAmount","Collateral","DebttoIncome"
from pathlib import Path
import logging
import math
import sys

import xgboost as xgb
import pandas as pd
ML_MODEL_DIR = Path(__file__).resolve().parent.parent / "ML_model"
if str(ML_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(ML_MODEL_DIR))

from explainer import get_shap_values
from explanation_engine import ExplanationEngine
from .gemini_explainer import explain_with_gemini
from .explanation_context import (
    FEATURE_LABELS,
    build_applicant_features,
    build_fallback_explanation,
    calculate_relationships,
    contribution_strengths,
    select_relevant_contributions,
)

model = xgb.Booster()
model_path = ML_MODEL_DIR / "Prediction_model.json"
model.load_model(model_path)

FEATURES = [
    "Age",
    "AnnualIncome",
    "CreditScore",
    "EmploymentStatus",
    "LoanAmount",
    "Collateral",
    "DebttoIncome"
]
logger = logging.getLogger(__name__)
if model.feature_names and model.feature_names != FEATURES:
    raise RuntimeError("Saved model features do not match the inference feature order")


def user_input():
    
    age = int(input("Enter your age: "))
    income = int(input("Enter your annual income: "))
    credit_score = int(input("Enter your credit score: "))
    employment_status = input("Enter Employment Status(Employed,Self-Employed,Unemployed): ")
    loan_amount = int(input("Enter loan amount needed: "))
    collateral = int(input("Enter the value of the collateral: "))
    debt = int(input("Amount of debt currently: "))
    
    enc_emp_status = {
        "Employed" : 0,
        "Self-Employed": 1,
        "Unemployed": 2
    }[employment_status]
    
    monthly_income = income / 12
    debt_to_income = debt / monthly_income if monthly_income else 0
    
    input_values = [age,income,credit_score,enc_emp_status,loan_amount,collateral,debt_to_income]
    
    return input_values
     
def predict(input_values):
   
    data = pd.DataFrame([input_values], columns=[
        "Age",
        "AnnualIncome",
        "CreditScore",
        "EmploymentStatus",
        "LoanAmount",
        "Collateral",
        "DebttoIncome"
    ])
    
    data_matrix = xgb.DMatrix(data)
    
    
    prediction = model.predict(data_matrix)
    
    return prediction[0]


def predict_applicant(age, income, credit_score, employment_status, loan_amount, collateral, monthly_debt):
    """Return the model prediction and its contextual local explanation."""
    employment_codes = {
        "Employed": 0,
        "Self-Employed": 1,
        "Unemployed": 2,
    }
    monthly_income = income / 12
    debt_to_income = monthly_debt / monthly_income if monthly_income else 0
    input_values = [
        age,
        income,
        credit_score,
        employment_codes[employment_status],
        loan_amount,
        collateral,
        debt_to_income,
    ]

    probability = float(predict(input_values))
    predicted_class = int(probability >= 0.5)
    decision = "approved" if predicted_class == 1 else "rejected"

    raw_values = {
        "Age": age,
        "AnnualIncome": income,
        "CreditScore": credit_score,
        "EmploymentStatus": employment_status,
        "LoanAmount": loan_amount,
        "Collateral": collateral,
        "DebttoIncome": debt_to_income,
        "MonthlyDebtPayments": monthly_debt,
    }
    applicant_features = build_applicant_features(raw_values)
    feature_value_lookup = {item["feature"]: item["value"] for item in applicant_features}
    relationships = calculate_relationships(income, loan_amount, monthly_debt)

    all_contributions = []
    try:
        shap_values = get_shap_values(model, input_values, predicted_class=predicted_class)
        if len(shap_values) != len(FEATURES):
            raise ValueError("SHAP feature count differs from the model")
        all_contributions = [
            {
                "feature": FEATURE_LABELS[name],
                "value": feature_value_lookup[FEATURE_LABELS[name]],
                "impact": float(impact),
            }
            for name, impact in zip(FEATURES, shap_values)
            if math.isfinite(float(impact))
        ]
    except Exception as exc:
        logger.warning("Local feature attribution failed (%s).", type(exc).__name__)

    selected = select_relevant_contributions(all_contributions)
    strengths = contribution_strengths(all_contributions)
    for item in selected:
        item["relative_strength"] = strengths[item["feature"]]

    net_impact = math.fsum(item["impact"] for item in all_contributions)
    combined_direction = (
        "toward_predicted_class" if net_impact > 0
        else "away_from_predicted_class" if net_impact < 0
        else "balanced"
    )
    narrative = explain_with_gemini(
        decision=decision,
        applicant_features=applicant_features,
        contributions=selected,
        relationships=relationships,
        combined_direction=combined_direction,
    )

    if narrative is not None:
        explanations_by_feature = narrative["feature_explanations"]
        supporting, opposing = [], []
        for item in selected:
            feature_explanation = {
                "feature": item["feature"],
                "value": item["value"],
                "relative_strength": item["relative_strength"],
                "explanation": explanations_by_feature[item["feature"]],
            }
            (supporting if item["impact"] > 0 else opposing).append(feature_explanation)
        explanation = {
            "summary": (
                "The model estimates loan approval."
                if decision == "approved" else "The model estimates loan rejection."
            ),
            "supporting": supporting,
            "opposing": opposing,
            "relationships": relationships,
            "overall": narrative["overall_reasoning"],
        }
    else:
        explanation = build_fallback_explanation(
            decision, selected, relationships, all_contributions=all_contributions
        )

    return {
        "probability": probability,
        "decision": decision,
        "explanation": explanation,
        # Keep the former flat explanation key for existing API clients.
        "explanations": [
            f"{item['feature']}: {item['explanation']}"
            for item in explanation["supporting"] + explanation["opposing"]
        ],
    }

    
def main():
    
    data = user_input()
    prediction = predict(data)
    predicted_class = int(float(prediction) >= 0.5)
    shap_values = get_shap_values(model, data, predicted_class=predicted_class)
    engine = ExplanationEngine(FEATURES)
    explanations = engine.generate(data, shap_values)
    
    print("\n========================")
    print("LOAN DECISION")
    print("========================")
    
    if prediction >= 0.5:
        print("Loan Approved")
    else:
        print("Loan Rejected")
        
        
    
    print("\nWhy?") 
    
    for explanation in explanations:
        print("->", explanation)
    
if __name__ == "__main__":
    main()
    
    
