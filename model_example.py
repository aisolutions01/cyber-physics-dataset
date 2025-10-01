# =========================
# model_example.py
# =========================
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


from data_generator import IncidentGenerator


def train_model(n_samples=50):
gen = IncidentGenerator()
data = [gen.generate_incident() for _ in range(n_samples)]
df = pd.DataFrame(data)


# simple label: classify if incident is critical (High severity)
df["label"] = df["severity"].apply(lambda x: 1 if x == "High" else 0)


pipeline = Pipeline([
("vec", CountVectorizer()),
("clf", LogisticRegression())
])


pipeline.fit(df["incident_type"], df["label"])
return pipeline, df


if __name__ == "__main__":
model, dataset = train_model()
print(dataset.head())
print("Model trained on-the-fly!")
