"""
Load testing for RAG-Bench API using Locust.

Usage:
    # Web UI (http://localhost:8089):
    python -m locust --host https://localhost

    # Headless — smoke test (5 users, 2 minutes):
    python -m locust --host https://localhost --headless -u 5 -r 1 -t 2m

    # Headless — sustained load (15 users, 5 minutes):
    python -m locust --host https://localhost --headless -u 15 -r 3 -t 5m

    # Browse-only (no queries):
    python -m locust --host https://localhost --headless -u 20 -r 5 -t 2m BrowseUser

    # Query stress test only:
    python -m locust --host https://localhost --headless -u 6 -r 2 -t 3m QueryUser
"""

import random

import urllib3
from locust import HttpUser, between, task

# Suppress SSL warnings when testing against localhost with self-signed cert
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Realistic queries spanning different difficulty levels and topics
QUERIES = [
    "What is the transformer architecture?",
    "How does BERT handle masked language modeling?",
    "Compare GPT and BERT architectures",
    "What datasets are used for evaluating large language models?",
    "Explain attention mechanisms in neural networks",
    "What is retrieval augmented generation?",
    "How does LoRA work for fine-tuning?",
    "What are the key contributions of the ResNet paper?",
    "Explain the difference between encoder and decoder transformers",
    "What benchmarks measure reasoning in LLMs?",
    "How does chain-of-thought prompting improve performance?",
    "What is RLHF and how is it used in language models?",
    "Describe the architecture of diffusion models",
    "What are mixture of experts models?",
    "How does Flash Attention reduce memory usage?",
]


class BrowseUser(HttpUser):
    """Simulates users browsing the UI — health checks, stats, papers.

    These are fast endpoints that should never block. Weight=5 means 5x more
    browse users than query users when running all user types together.
    """

    weight = 5
    wait_time = between(1, 3)

    def on_start(self):
        self.client.verify = False

    @task(10)
    def health(self):
        self.client.get("/api/health")

    @task(8)
    def stats(self):
        self.client.get("/api/stats")

    @task(5)
    def queue_status(self):
        self.client.get("/api/queue/status")

    @task(3)
    def metrics_summary(self):
        self.client.get("/api/metrics/summary")

    @task(4)
    def list_papers(self):
        self.client.get("/api/papers")

    @task(3)
    def graph_context(self):
        q = random.choice(["transformer", "BERT", "GPT", "attention", "LoRA"])
        self.client.get(f"/api/graph/context?question={q}")


class QueryUser(HttpUser):
    """Simulates users asking questions — hits LLM + ChromaDB + reranker.

    Each query takes ~35s. Server allows 1 concurrent + 4 queued, then returns
    429. Weight=1 means far fewer query users than browse users.
    """

    weight = 1
    wait_time = between(5, 15)

    def on_start(self):
        self.client.verify = False

    @task
    def query(self):
        with self.client.post(
            "/api/query",
            json={
                "question": random.choice(QUERIES),
                "top_k": 5,
            },
            timeout=120,
            catch_response=True,
        ) as response:
            if response.status_code == 429:
                # Server at capacity — expected under load, not a failure
                response.success()
