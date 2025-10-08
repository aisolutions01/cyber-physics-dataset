# =========================
# data_generator.py
# =========================
import json
import time
from datetime import datetime
from typing import Generator, Dict, Any


class IncidentGenerator:
    def __init__(self, data_path: str = "streams_1k.json"):
        """Initialize generator by loading the pre-generated streaming JSON file."""
        with open(data_path, "r") as f:
            self.events = json.load(f)
        self.index = 0
        self.total = len(self.events)

    def generate_incident(self) -> Dict[str, Any]:
        """Return one incident from the loaded JSON, emulating on-the-fly behavior."""
        if self.index >= self.total:
            # restart or stop when reaching the end
            self.index = 0
        event = self.events[self.index]
        self.index += 1

        # normalize timestamp if needed
        if "timestamp" not in event:
            event["timestamp"] = datetime.utcnow().isoformat()

        return event

    def stream(self, n: int = 10, delay: float = 0.5) -> Generator[Dict[str, Any], None, None]:
        """Yield n incidents sequentially, with optional delay to simulate streaming."""
        for _ in range(min(n, self.total)):
            yield self.generate_incident()
            time.sleep(delay)


# Example usage:
if __name__ == "__main__":
    gen = IncidentGenerator("streams_1k.json")
    for evt in gen.stream(n=5, delay=0.1):
        print(evt)
