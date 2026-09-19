# "Age","AnnualIncome","CreditScore","EmploymentStatus","LoanAmount","Collateral","DebttoIncome"
from pathlib import Path
import xgboost as xgb
import pandas as pd


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
    model = xgb.Booster()
    
    model_path = Path(__file__).parent.parent /"ML_model" /"Prediction_model.json"
    model.load_model(model_path)
    
    
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
    print(f"Loan Status: {predict(user_input())}")
    
if __name__ == "__main__":
    main()
    
    