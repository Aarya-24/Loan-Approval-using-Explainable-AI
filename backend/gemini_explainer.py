"""Generate a grounded, structured explanation from local model outputs."""

import json
import logging
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

logger = logging.getLogger(__name__)


class FeatureNarrative(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feature: str
    explanation: str


class GeminiNarrative(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feature_explanations: list[FeatureNarrative]
    overall_reasoning: str


SYSTEM_INSTRUCTION = """You translate this application's local feature attributions
into a clear, balanced explanation. The machine-learning model alone made the
decision; do not reconsider, recalculate, or change it. Use only the supplied
applicant values, feature attributions, class, and pre-calculated relationships.
Do not use external information or general lending rules.

Explain every supplied selected feature exactly once. For each, name the actual
value, explain a relevant relationship to another supplied value when one is
available, and say whether its local attribution supports or opposes this
prediction. The supplied direction and relative strength are authoritative;
never invent or reverse either. A calculated ratio is arithmetic context, not
proof of affordability or repayment ability. Do not infer missing loan terms,
interest rates, credit history, or job tenure. Do not claim that a feature
caused the decision.

Describe relative influence using only the supplied rank (strongest, next
strongest, etc.). SHAP values here are raw log-odds for the predicted class;
they are not probabilities or percentage-point changes. Do not show SHAP
numbers, log-odds, probabilities, or a baseline in the prose. Explain both
supporting and opposing features where present. The overall reasoning must
state that the final result reflects the combined factors and must agree with
the supplied predicted class and net contribution direction. If the data does
not justify a specific interpretation, say the model's local attribution
shows the direction but the available inputs do not explain a more specific
reason.

Return only the requested JSON object. Do not add keys or omit a selected
feature."""


def _has_forbidden_prediction_detail(text):
    return bool(re.search(
        r"\b(probability|chance|likelihood|confidence|odds|baseline|log[- ]odds|SHAP)\b",
        text,
        re.I,
    ))


def validate_gemini_narrative(raw_text, selected_features):
    """Parse Gemini JSON and reject missing, invented, or mismatched features."""
    response = GeminiNarrative.model_validate_json(raw_text)
    expected = {item["feature"] for item in selected_features}
    actual = [item.feature for item in response.feature_explanations]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Gemini did not explain exactly the selected features")
    if any(not item.explanation.strip() for item in response.feature_explanations):
        raise ValueError("Gemini returned an empty feature explanation")
    prose = [item.explanation for item in response.feature_explanations]
    prose.append(response.overall_reasoning)
    if not response.overall_reasoning.strip() or any(
        _has_forbidden_prediction_detail(item) for item in prose
    ):
        raise ValueError("Gemini returned invalid or disallowed explanation text")
    return response


def explain_with_gemini(
    *,
    decision,
    applicant_features,
    contributions,
    relationships,
    combined_direction,
):
    """Return structured Gemini feature text, or None to trigger local fallback."""
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key or not contributions:
        return None

    try:
        from google import genai
        from google.genai import types

        strengths = {
            item["feature"]: item["relative_strength"] for item in contributions
        }
        prompt_data = {
            "predicted_class": decision,
            "applicant_features": applicant_features,
            "calculated_relationships": relationships,
            "selected_local_feature_attributions": [
                {
                    "feature": item["feature"],
                    "applicant_value": item["value"],
                    "shap_value": item["impact"],
                    "output_scale": "raw log-odds for the predicted class",
                    "direction": "supports_prediction" if item["impact"] > 0 else "opposes_prediction",
                    "relative_strength": strengths[item["feature"]],
                }
                for item in contributions
            ],
            "combined_local_contribution_direction": combined_direction,
        }
        model_name = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
        thinking_level = "minimal" if model_name.startswith("gemini-3.6") else "low"
        client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=15_000),
        )
        response = client.models.generate_content(
            model=model_name,
            contents=json.dumps(prompt_data, ensure_ascii=False),
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=0.2,
                thinking_config=types.ThinkingConfig(thinking_level=thinking_level),
                max_output_tokens=1400,
                response_mime_type="application/json",
                response_schema=GeminiNarrative,
            ),
        )
        finish_reason = (
            response.candidates[0].finish_reason.name
            if response.candidates and response.candidates[0].finish_reason
            else None
        )
        if finish_reason == "MAX_TOKENS":
            logger.warning("Gemini explanation reached its output token limit.")
            return None
        narrative = validate_gemini_narrative(response.text or "{}", contributions)
        by_feature = {item.feature: item.explanation for item in narrative.feature_explanations}
        return {
            "feature_explanations": by_feature,
            "overall_reasoning": narrative.overall_reasoning.strip(),
        }
    except Exception as exc:
        # Do not log exception text: SDK errors can contain request details.
        logger.warning("Gemini explanation failed (%s); using local fallback.", type(exc).__name__)
        return None
