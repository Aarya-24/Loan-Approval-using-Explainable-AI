"""Prepare grounded applicant context and a safe local fallback explanation."""

import math

FEATURE_LABELS = {
    "Age": "Age",
    "AnnualIncome": "Annual income",
    "CreditScore": "Credit score",
    "EmploymentStatus": "Employment status",
    "LoanAmount": "Requested loan amount",
    "Collateral": "Collateral value",
    "DebttoIncome": "Debt-to-income ratio",
    "MonthlyDebtPayments": "Current monthly debt payments",
}
FEATURE_DEFINITIONS = {
    "Age": "Applicant age in years.",
    "AnnualIncome": "Applicant-stated income over a year; gross or net status is not supplied.",
    "CreditScore": "The credit score value entered by the applicant.",
    "EmploymentStatus": "The selected employment category; no job tenure is collected.",
    "LoanAmount": "The requested loan principal amount.",
    "Collateral": "The entered value of collateral, if any.",
    "DebttoIncome": "Monthly debt payments divided by monthly income.",
    "MonthlyDebtPayments": "Applicant-stated recurring debt payments per month.",
}


def format_currency(value):
    number = float(value)
    formatted = f"{number:,.2f}".rstrip("0").rstrip(".")
    return f"${formatted}"


def build_applicant_features(values):
    """Map model-ready values to readable values without changing model inputs."""
    readable = []
    for name, value in values.items():
        if value is None:
            display_value = "Not provided"
        elif name in {"AnnualIncome", "LoanAmount", "Collateral", "MonthlyDebtPayments"}:
            display_value = format_currency(value)
        elif name == "Age":
            display_value = f"{int(value)} years"
        elif name == "DebttoIncome":
            display_value = f"{float(value):.1%}"
        else:
            display_value = value
        readable.append({
            "feature": FEATURE_LABELS.get(name, name),
            "value": display_value,
            "definition": FEATURE_DEFINITIONS.get(name, "Meaning not defined by the application."),
        })
    return readable


def calculate_relationships(annual_income, loan_amount, monthly_debt):
    """Calculate only same-currency and same-period ratios supported by inputs."""
    relationships = []
    annual_income = float(annual_income)
    loan_amount = float(loan_amount)
    monthly_debt = float(monthly_debt)

    if annual_income > 0 and loan_amount >= 0:
        ratio = loan_amount / annual_income
        relationships.append({
            "name": "Requested amount relative to annual income",
            "value": f"{ratio:.1%}",
            "calculation": (
                f"{format_currency(loan_amount)} requested / "
                f"{format_currency(annual_income)} annual income"
            ),
            "limitation": "This comparison alone does not establish affordability.",
        })

    monthly_income = annual_income / 12
    if monthly_income > 0 and monthly_debt >= 0:
        ratio = monthly_debt / monthly_income
        relationships.append({
            "name": "Monthly debt payments relative to monthly income",
            "value": f"{ratio:.1%}",
            "calculation": (
                f"{format_currency(monthly_debt)} monthly debt / "
                f"{format_currency(monthly_income)} monthly income"
            ),
            "limitation": "Loan term and interest rate were not provided.",
        })
    return relationships


def select_relevant_contributions(contributions, max_features=4):
    """Keep the largest local contributions and retain both directions."""
    valid = [
        item for item in contributions
        if item.get("value") is not None
        and isinstance(item.get("impact"), (int, float))
        and math.isfinite(float(item["impact"]))
        and float(item["impact"]) != 0
    ]
    ordered = sorted(valid, key=lambda item: abs(float(item["impact"])), reverse=True)
    if len(ordered) <= max_features:
        return ordered

    selected = []
    for sign in (1, -1):
        directional = [
            item for item in ordered
            if math.copysign(1, item["impact"]) == sign
        ]
        selected.extend(directional[: min(2, max_features // 2)])
    for item in ordered:
        if item not in selected and len(selected) < max_features:
            selected.append(item)
    return sorted(selected, key=lambda item: abs(float(item["impact"])), reverse=True)


def contribution_strengths(contributions):
    """Describe relative rank without treating SHAP values as probabilities."""
    ordered = sorted(contributions, key=lambda item: abs(float(item["impact"])), reverse=True)
    labels = ["strongest factor", "next strongest factor", "third strongest factor", "fourth strongest factor"]
    strengths = {}
    for rank, item in enumerate(ordered):
        strengths[item["feature"]] = labels[rank] if rank < len(labels) else "a smaller factor"
    return strengths


def build_fallback_explanation(
    decision, contributions, relationships, all_contributions=None
):
    """Create a structured, model-grounded explanation if Gemini is unavailable."""
    strength_source = all_contributions if all_contributions is not None else contributions
    strengths = contribution_strengths(strength_source)
    relationship_by_name = {item["name"]: item for item in relationships}
    supporting = []
    opposing = []
    for item in contributions:
        supports = item["impact"] > 0
        predicted_outcome = "approval" if decision == "approved" else "rejection"
        movement = (
            f"in support of the model's {predicted_outcome} estimate"
            if supports else f"against the model's {predicted_outcome} estimate"
        )
        relation = None
        if item["feature"] in {"Requested loan amount", "Annual income"}:
            relation = relationship_by_name.get("Requested amount relative to annual income")
        elif item["feature"] == "Debt-to-income ratio":
            relation = relationship_by_name.get("Monthly debt payments relative to monthly income")

        context = f" {relation['name']} is {relation['value']} ({relation['calculation']})." if relation else ""
        note = (
            f"The model treated {item['feature'].lower()} at {item['value']} as "
            f"the {strengths[item['feature']]}.{context} Its local contribution moved "
            f"the estimate {movement}; this describes the model's attribution, not a cause."
        )
        (supporting if supports else opposing).append({
            "feature": item["feature"],
            "value": item["value"],
            "relative_strength": strengths[item["feature"]],
            "explanation": note,
        })

    all_impacts = strength_source
    net_impact = math.fsum(float(item["impact"]) for item in all_impacts)
    if not all_impacts:
        overall = "A local feature explanation was unavailable, but the model's prediction is still shown."
    elif net_impact > 0:
        overall = "Taken together, the available feature effects supported the model's predicted outcome."
    elif net_impact < 0:
        overall = "Taken together, the available feature effects worked against the predicted outcome; the model still returned the displayed decision."
    else:
        overall = "The available feature effects balanced each other; the model returned the displayed decision based on its full calculation."

    return {
        "summary": "The model estimates loan approval." if decision == "approved" else "The model estimates loan rejection.",
        "supporting": supporting,
        "opposing": opposing,
        "relationships": relationships,
        "overall": overall,
    }
