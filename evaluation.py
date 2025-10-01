# =========================
# evaluation.py
# =========================
from sklearn.metrics import classification_report


def evaluate_model(model, n_samples=20):
gen = IncidentGenerator()
test_data = [gen.generate_incident() for _ in range(n_samples)]
df = pd.DataFrame(test_data)
df["label"] = df["severity"].apply(lambda x: 1 if x == "High" else 0)


preds = model.predict(df["incident_type"])
print(classification_report(df["label"], preds))
