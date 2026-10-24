import json
import os
import unittest
from unittest.mock import patch

import numpy as np

from ML_model.explainer import select_class_shap_values
from backend.explanation_context import (
    build_applicant_features,
    build_fallback_explanation,
    calculate_relationships,
    select_relevant_contributions,
)
from backend.gemini_explainer import explain_with_gemini, validate_gemini_narrative
from backend import predict as prediction_module


class ClassSpecificShapTests(unittest.TestCase):
    def test_binary_logistic_shap_is_oriented_toward_rejected_class(self):
        raw_approval_contributions = np.array([[0.3, -0.2, 0.1]])
        rejected = select_class_shap_values(
            raw_approval_contributions,
            predicted_class=0,
            feature_count=3,
            objective="binary:logistic",
        )
        self.assertEqual(rejected, [-0.3, 0.2, -0.1])

    def test_multiclass_shap_selects_predicted_output(self):
        values = np.array([[[0.1, 0.2, 0.3], [1.1, 1.2, 1.3]]])
        selected = select_class_shap_values(
            values, predicted_class=2, feature_count=2, objective="multi:softprob"
        )
        self.assertEqual(selected, [0.3, 1.3])

    def test_invalid_or_nonfinite_shap_values_are_rejected(self):
        with self.assertRaises(ValueError):
            select_class_shap_values(
                [[float("nan"), 0.2]], 1, 2, objective="binary:logistic"
            )

    def test_approved_class_keeps_positive_class_shap_direction(self):
        approved = select_class_shap_values(
            [[0.3, -0.2]], 1, 2, objective="binary:logistic"
        )
        self.assertEqual(approved, [0.3, -0.2])


class ContextAndSelectionTests(unittest.TestCase):
    def test_contextual_ratios_use_compatible_periods(self):
        ratios = calculate_relationships(50_000, 5_000, 400)
        self.assertEqual(ratios[0]["value"], "10.0%")
        self.assertEqual(ratios[1]["value"], "9.6%")
        self.assertIn("does not establish affordability", ratios[0]["limitation"])

    def test_employment_status_is_readable_and_ratio_is_formatted(self):
        values = build_applicant_features({"EmploymentStatus": "Self-Employed", "DebttoIncome": 0.096})
        self.assertEqual(values[0]["value"], "Self-Employed")
        self.assertEqual(values[1]["value"], "9.6%")

    def test_selection_retains_strong_factors_from_both_directions(self):
        contributions = [
            {"feature": f"f{i}", "value": i, "impact": impact}
            for i, impact in enumerate([0.9, 0.8, 0.7, 0.6, -0.5, -0.4, -0.3])
        ]
        selected = select_relevant_contributions(contributions, max_features=4)
        self.assertEqual({item["feature"] for item in selected}, {"f0", "f1", "f4", "f5"})

    def test_fallback_still_returns_structured_prediction_explanation(self):
        items = [
            {"feature": "Requested loan amount", "value": "$5,000", "impact": 0.5,
             "relative_strength": "strongest factor"},
            {"feature": "Annual income", "value": "$50,000", "impact": -0.2,
             "relative_strength": "next strongest factor"},
        ]
        fallback = build_fallback_explanation(
            "approved", items, calculate_relationships(50_000, 5_000, 0)
        )
        self.assertEqual(len(fallback["supporting"]), 1)
        self.assertEqual(len(fallback["opposing"]), 1)
        self.assertIn("10.0%", fallback["supporting"][0]["explanation"])


class StructuredGeminiTests(unittest.TestCase):
    def setUp(self):
        self.selected = [
            {"feature": "Requested loan amount", "value": "$5,000", "impact": 0.4,
             "relative_strength": "strongest factor"},
            {"feature": "Annual income", "value": "$50,000", "impact": -0.2,
             "relative_strength": "next strongest factor"},
        ]

    def test_structured_response_requires_exact_feature_set(self):
        raw = json.dumps({
            "feature_explanations": [
                {"feature": "Requested loan amount", "explanation": "It is 10% of annual income."},
                {"feature": "Annual income", "explanation": "It provides context for the requested amount."},
            ],
            "overall_reasoning": "Both factors contributed to the model's estimate.",
        })
        parsed = validate_gemini_narrative(raw, self.selected)
        self.assertEqual(len(parsed.feature_explanations), 2)

        hallucinated = json.loads(raw)
        hallucinated["feature_explanations"][0]["feature"] = "Credit history"
        with self.assertRaises(ValueError):
            validate_gemini_narrative(json.dumps(hallucinated), self.selected)
        with self.assertRaises(ValueError):
            validate_gemini_narrative("{invalid json", self.selected)

    def test_probability_or_baseline_language_is_rejected(self):
        raw = json.dumps({
            "feature_explanations": [
                {"feature": "Requested loan amount", "explanation": "This raised the probability."},
                {"feature": "Annual income", "explanation": "It was considered with loan amount."},
            ],
            "overall_reasoning": "These factors combined.",
        })
        with self.assertRaises(ValueError):
            validate_gemini_narrative(raw, self.selected)

    def test_gemini_api_failure_returns_fallback_signal_without_logging_input(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "unit-test-key", "GEMINI_MODEL": "gemini-3.6-flash"}):
            with patch("google.genai.Client", side_effect=RuntimeError("sensitive data $50,000")):
                with self.assertLogs("backend.gemini_explainer", level="WARNING") as logs:
                    result = explain_with_gemini(
                        decision="approved",
                        applicant_features=[],
                        contributions=self.selected,
                        relationships=[],
                        combined_direction="toward_predicted_class",
                    )
        self.assertIsNone(result)
        self.assertNotIn("$50,000", " ".join(logs.output))
        self.assertNotIn("unit-test-key", " ".join(logs.output))


class PredictionPipelineTests(unittest.TestCase):
    @patch("backend.predict.explain_with_gemini", return_value=None)
    @patch("backend.predict.get_shap_values", return_value=[0.5, -0.4, 0.3, 0.1, -0.2, 0.0, 0.15])
    @patch("backend.predict.predict", return_value=0.63)
    def test_gemini_failure_keeps_model_decision_and_structured_fallback(
        self, mock_predict, mock_shap, mock_gemini
    ):
        result = prediction_module.predict_applicant(
            age=32,
            income=50_000,
            credit_score=700,
            employment_status="Employed",
            loan_amount=5_000,
            collateral=0,
            monthly_debt=400,
        )
        self.assertEqual(result["decision"], "approved")
        self.assertEqual(result["probability"], 0.63)
        self.assertEqual(result["explanation"]["summary"], "The model estimates loan approval.")
        self.assertTrue(result["explanation"]["supporting"])
        self.assertTrue(result["explanation"]["opposing"])
        self.assertTrue(result["explanation"]["relationships"])
        mock_shap.assert_called_once()
        self.assertEqual(mock_shap.call_args.kwargs["predicted_class"], 1)
        mock_gemini.assert_called_once()
        mock_predict.assert_called_once()


if __name__ == "__main__":
    unittest.main()
