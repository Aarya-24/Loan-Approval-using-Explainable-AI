import shap
import pandas as pd

FEATURES = [
    "Age",
    "AnnualIncome",
    "CreditScore",
    "EmploymentStatus",
    "LoanAmount",
    "Collateral",
    "DebttoIncome"
]
def get_shap_values(model,applicant):
    explainer = shap.TreeExplainer(model)
    

    applicant_df = pd.DataFrame(
        [applicant],
        columns=FEATURES
    )

    shap_values = explainer.shap_values(applicant_df)

    return shap_values