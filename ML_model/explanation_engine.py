class ExplanationEngine:

    def __init__(self, feature_names):
        self.feature_names = feature_names

    def generate(self, applicant, shap_values, top_n=5):

        feature_impacts = []

        for feature, value, impact in zip(
            self.feature_names,
            applicant,
            shap_values
        ):
            feature_impacts.append({
                "feature": feature,
                "value": value,
                "impact": impact
            })

        # Most influential features first
        feature_impacts.sort(
            key=lambda x: abs(x["impact"]),
            reverse=True
        )

        explanations = []

        for item in feature_impacts[:top_n]:

            feature = item["feature"]
            value = item["value"]
            impact = item["impact"]

            text = self.explain_feature(
                feature,
                value,
                impact
            )

            explanations.append(text)

        return explanations

    def explain_feature(self, feature, value, impact):

        direction = "positively" if impact > 0 else "negatively"

        return (
            f"{feature} ({value}) "
            f"{direction} influenced the loan decision."
        )