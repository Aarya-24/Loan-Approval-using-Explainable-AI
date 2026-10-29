const form = document.querySelector("#prediction-form");
const submitButton = form.querySelector("button[type='submit']");
const errorBox = document.querySelector("#form-error");
const resultPanel = document.querySelector("#result-panel");

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  errorBox.hidden = true;
  resultPanel.hidden = true;

  const formData = new FormData(form);
  const payload = {
    name: formData.get("name").trim(),
    age: Number(formData.get("age")),
    annual_income: Number(formData.get("annual_income")),
    credit_score: Number(formData.get("credit_score")),
    employment_status: formData.get("employment_status"),
    loan_amount: Number(formData.get("loan_amount")),
    collateral: Number(formData.get("collateral")),
    monthly_debt: Number(formData.get("monthly_debt")),
  };

  submitButton.disabled = true;
  submitButton.querySelector(".button-label").textContent = "Calculating your estimate…";

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const result = await response.json();
    if (!response.ok) {
      const detail = Array.isArray(result.detail)
        ? result.detail.map((item) => item.msg).join(" ")
        : result.detail;
      throw new Error(detail || "Please check the details you entered.");
    }
    showResult(result);
  } catch (error) {
    errorBox.textContent = error instanceof TypeError
      ? "We couldn’t reach the prediction service. Please check that the backend is running and try again."
      : error.message;
    errorBox.hidden = false;
  } finally {
    submitButton.disabled = false;
    submitButton.querySelector(".button-label").textContent = "See my estimate";
  }
});

function showResult(result) {
  const approved = result.decision === "approved";
  const percent = Math.round(result.probability * 100);
  const resultIcon = document.querySelector("#result-icon");
  resultIcon.textContent = approved ? "✓" : "↗";
  resultIcon.classList.toggle("rejected", !approved);
  document.querySelector("#result-title").textContent = approved
    ? "The model estimates approval"
    : "The model estimates rejection";
  document.querySelector("#result-probability").textContent = `${percent}%`;
  document.querySelector("#result-request-id").textContent = result.loan_request_id;
  document.querySelector("#result-summary").textContent = approved
    ? "Based on the details you provided, the estimate leans toward approval."
    : "Based on the details you provided, the estimate leans toward rejection.";
  document.querySelector("#result-meter-fill").style.width = `${percent}%`;

  const structuredExplanation = result.explanation && typeof result.explanation === "object"
    ? result.explanation
    : {};
  const hasStructuredExplanation = Boolean(
    structuredExplanation.summary || structuredExplanation.overall ||
    (structuredExplanation.supporting || []).length ||
    (structuredExplanation.opposing || []).length ||
    (structuredExplanation.relationships || []).length
  );
  const legacyExplanations = hasStructuredExplanation
    ? []
    : Array.isArray(result.explanations)
      ? result.explanations
      : typeof result.explanation === "string"
        ? [result.explanation]
        : [];
  const explanation = structuredExplanation;
  document.querySelector("#explanation-summary").textContent = explanation.summary || "";
  document.querySelector("#explanation-overall").textContent = explanation.overall || "";
  renderFeatureGroup("supporting-group", "supporting-list", explanation.supporting || []);
  renderFeatureGroup("opposing-group", "opposing-list", explanation.opposing || []);
  renderRelationshipGroup(explanation.relationships || []);
  renderLegacyExplanations(legacyExplanations);

  const hasExplanation = Boolean(
    explanation.summary || explanation.overall ||
    (explanation.supporting || []).length ||
    (explanation.opposing || []).length ||
    (explanation.relationships || []).length ||
    legacyExplanations.length
  );
  document.querySelector("#explanation-unavailable").hidden = hasExplanation;

  resultPanel.hidden = false;
  resultPanel.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function renderLegacyExplanations(explanations) {
  const group = document.querySelector("#legacy-explanation-group");
  const list = document.querySelector("#legacy-explanation-list");
  list.replaceChildren();
  group.hidden = explanations.length === 0;
  explanations.forEach((explanation) => {
    const item = document.createElement("li");
    item.textContent = explanation;
    list.append(item);
  });
}

function renderFeatureGroup(groupId, listId, features) {
  const group = document.querySelector(`#${groupId}`);
  const list = document.querySelector(`#${listId}`);
  list.replaceChildren();
  group.hidden = features.length === 0;

  features.forEach((feature) => {
    const item = document.createElement("li");
    item.className = "explanation-item";
    const title = document.createElement("strong");
    title.textContent = `${feature.feature}: ${feature.value}`;
    const strength = document.createElement("small");
    strength.textContent = feature.relative_strength;
    const detail = document.createElement("p");
    detail.textContent = feature.explanation;
    item.append(title, strength, detail);
    list.append(item);
  });
}

function renderRelationshipGroup(relationships) {
  const group = document.querySelector("#relationships-group");
  const list = document.querySelector("#relationships-list");
  list.replaceChildren();
  group.hidden = relationships.length === 0;

  relationships.forEach((relationship) => {
    const item = document.createElement("li");
    const title = document.createElement("strong");
    title.textContent = `${relationship.name}: ${relationship.value}`;
    const detail = document.createElement("span");
    detail.textContent = `${relationship.calculation}. ${relationship.limitation}`;
    item.append(title, detail);
    list.append(item);
  });
}

document.querySelector("#try-again").addEventListener("click", () => {
  resultPanel.hidden = true;
  form.querySelector("input").focus();
});

async function lookupRequest(form, resultBox, decisionOnly) {
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    resultBox.textContent = "Looking up request…";
    const id = new FormData(form).get("loan_request_id").trim().toUpperCase();
    try {
      const response = await fetch(`/api/loan-requests/${encodeURIComponent(id)}`);
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Could not retrieve this request.");
      resultBox.replaceChildren();
      if (decisionOnly) {
        const summary = document.createElement("p");
        summary.textContent = `${data.name}: ${data.decision.toUpperCase()} · ${Math.round(data.probability * 100)}% estimated approval probability`;
        resultBox.append(summary);
      } else {
        const list = document.createElement("dl");
        const labels = { loan_request_id: "Loan request ID", name: "Name", age: "Age", annual_income: "Annual income", credit_score: "Credit score", employment_status: "Employment status", loan_amount: "Loan amount", collateral: "Collateral", monthly_debt: "Monthly debt payments", decision: "Decision", probability: "Approval probability", created_at: "Created" };
        Object.entries(labels).forEach(([key, label]) => {
          const term = document.createElement("dt"); term.textContent = label;
          const value = document.createElement("dd");
          value.textContent = key === "probability" ? `${Math.round(data[key] * 100)}%` : key === "annual_income" || key === "loan_amount" || key === "collateral" || key === "monthly_debt" ? new Intl.NumberFormat("en-IN", { style: "currency", currency: "INR" }).format(data[key]) : data[key];
          list.append(term, value);
        });
        resultBox.append(list);
      }
    } catch (error) { resultBox.textContent = error.message; }
  });
}

lookupRequest(document.querySelector("#full-lookup-form"), document.querySelector("#full-lookup-result"), false);
lookupRequest(document.querySelector("#decision-lookup-form"), document.querySelector("#decision-lookup-result"), true);
