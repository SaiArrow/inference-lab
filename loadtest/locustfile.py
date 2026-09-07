# loadtest/locustfile.py
from locust import HttpUser, task, between
import random

SAMPLES = [
    "an absolute delight from start to finish",
    "tedious, overlong, and painfully self-serious",
    "the cinematography carries an otherwise thin script",
    "i have never checked my watch more often",
]

class PredictUser(HttpUser):
    wait_time = between(0.05, 0.2)

    @task(9)
    def predict(self):
        self.client.post("/predict", json={"texts": [random.choice(SAMPLES)]})

    @task(1)
    def batch(self):
        self.client.post("/predict", json={"texts": random.sample(SAMPLES, 3)})


from locust import LoadTestShape

class StepLoad(LoadTestShape):
    # durations are cumulative elapsed seconds
    stages = [
        {"duration":  60, "users":  2,  "spawn_rate": 2},
        {"duration": 120, "users":  5,  "spawn_rate": 2},
        {"duration": 180, "users": 10,  "spawn_rate": 2},
        {"duration": 240, "users": 20,  "spawn_rate": 5},
        {"duration": 300, "users": 35,  "spawn_rate": 5},
        {"duration": 360, "users": 50,  "spawn_rate": 5},
    ]

    def tick(self):
        elapsed = self.get_run_time()
        for stage in self.stages:
            if elapsed < stage["duration"]:
                return stage["users"], stage["spawn_rate"]
        return None