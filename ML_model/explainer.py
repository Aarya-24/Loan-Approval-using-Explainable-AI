"""Calculate local SHAP values for the model class being explained."""

import json

import numpy as np
import pandas as pd
import shap

FEATURES = [
    "Age",
    "AnnualIncome",
    "CreditScore",
    "EmploymentStatus",
    "LoanAmount",
    "Collateral",
    "DebttoIncome",
]


def get_model_objective(model):
    """Read the XGBoost objective so output-space handling is explicit."""
    try:
        config = json.loads(model.save_config())
        return config["learner"]["objective"]["name"]
    except (AttributeError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def select_class_shap_values(
    shap_values, predicted_class, feature_count, objective="binary:logistic"
):
    """Select and orient one-row SHAP values for the requested model class.

    XGBoost's binary:logistic TreeExplainer output is one vector for class 1 in
    raw log-odds. For class 0, negating that vector gives contributions to the
    opposite log-odds, so positive values always support the predicted class.
    Multi-output arrays and older per-class lists are selected by class index.
    """
    if predicted_class < 0:
        raise ValueError("predicted_class must be non-negative")

    is_single_binary_output = False
    if isinstance(shap_values, list):
        if not shap_values:
            raise ValueError("SHAP returned no class outputs")
        if len(shap_values) == 1:
            selected = np.asarray(shap_values[0])
            is_single_binary_output = objective == "binary:logistic"
        else:
            if predicted_class >= len(shap_values):
                raise ValueError("Predicted class has no corresponding SHAP output")
            selected = np.asarray(shap_values[predicted_class])
    else:
        values = np.asarray(shap_values)
        if values.ndim == 3:
            if values.shape[1] == feature_count:  # (samples, features, outputs)
                output_count = values.shape[2]
                if output_count == 1:
                    selected = values[0, :, 0]
                    is_single_binary_output = objective == "binary:logistic"
                else:
                    if predicted_class >= output_count:
                        raise ValueError("Predicted class has no corresponding SHAP output")
                    selected = values[0, :, predicted_class]
            elif values.shape[2] == feature_count:  # (samples, outputs, features)
                output_count = values.shape[1]
                if output_count == 1:
                    selected = values[0, 0, :]
                    is_single_binary_output = objective == "binary:logistic"
                else:
                    if predicted_class >= output_count:
                        raise ValueError("Predicted class has no corresponding SHAP output")
                    selected = values[0, predicted_class, :]
            else:
                raise ValueError("SHAP output does not match the model feature count")
        elif values.ndim == 2:
            if values.shape == (1, feature_count):
                selected = values[0]
                is_single_binary_output = objective == "binary:logistic"
            elif values.shape == (feature_count, 1):
                selected = values[:, 0]
                is_single_binary_output = objective == "binary:logistic"
            elif values.shape[0] == 1 and values.shape[1] > feature_count:
                output_count = values.shape[1] // feature_count
                if values.shape[1] % feature_count or predicted_class >= output_count:
                    raise ValueError("Predicted class has no corresponding SHAP output")
                selected = values.reshape(1, feature_count, -1)[0, :, predicted_class]
            elif values.shape[0] == feature_count and values.shape[1] > 1:
                if predicted_class >= values.shape[1]:
                    raise ValueError("Predicted class has no corresponding SHAP output")
                selected = values[:, predicted_class]
            else:
                raise ValueError("SHAP output does not match the model feature count")
        elif values.ndim == 1 and values.shape[0] == feature_count:
            selected = values
            is_single_binary_output = objective == "binary:logistic"
        else:
            raise ValueError("SHAP output does not match the model feature count")

    selected = np.asarray(selected, dtype=float).reshape(-1)
    if selected.shape[0] != feature_count or not np.isfinite(selected).all():
        raise ValueError("SHAP returned missing or invalid feature contributions")
    if is_single_binary_output and predicted_class == 0:
        selected = -selected
    return selected.tolist()


def get_shap_values(model, applicant, predicted_class=1):
    """Return raw-log-odds SHAP values oriented toward the predicted class."""
    explainer = shap.TreeExplainer(model, model_output="raw")
    applicant_df = pd.DataFrame([applicant], columns=FEATURES)
    shap_values = explainer.shap_values(applicant_df)
    return select_class_shap_values(
        shap_values,
        predicted_class=predicted_class,
        feature_count=len(FEATURES),
        objective=get_model_objective(model),
    )
