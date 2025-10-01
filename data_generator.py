# =========================
# data_generator.py
# =========================
import random
import json
import time
from datetime import datetime


class IncidentGenerator:
def __init__(self, config_path="constraints.json"):
with open(config_path, "r") as f:
self.rules = json.load(f)


def generate_incident(self):
# pick random incident type from rules
incident_type = random.choice(self.rules["incident_types"])
severity = random.choice(self.rules["severity_levels"])
timestamp = datetime.utcnow().isoformat()


# simple policy-inspired constraint: severity depends on type
if incident_type == "Unauthorized Access":
severity = "High"


return {
"timestamp": timestamp,
"incident_type": incident_type,
"severity": severity,
"source": random.choice(["endpoint-1", "endpoint-2", "server-3"])
}


def stream(self, n=10, delay=1.0):
for _ in range(n):
yield self.generate_incident()
time.sleep(delay)
