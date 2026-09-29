# "Age","AnnualIncome","CreditScore","EmploymentStatus","LoanAmount","Collateral","DebttoIncome"
from pathlib import Path
import xgboost as xgb
import pandas as pd
from explainer import get_shap_values
from explanation_engine import ExplanationEngine

model = xgb.Booster()
    
model_path = Path(__file__).parent.parent /"ML_model" /"Prediction_model.json"
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
    
    debt_to_income = debt/income
    
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

    
def main():
    
    data = user_input()
    prediction = predict(data)
    
    #get shap values
    shap_values = get_shap_values(model,data)
    
    engine = ExplanationEngine(FEATURES)
    
    explanations = engine.generate(
        data,
        shap_values[0]
    )
    
    print("\n========================")
    print("LOAN DECISION")
    print("========================")
    
    if prediction == 1:
        print("Loan Approved")
    else:
        print("Loan Rejected")
        
        
    
    print("\nWhy?") 
    
    for explanation in explanations:
        print("•", explanation)
    
if __name__ == "__main__":
    main()
    
    