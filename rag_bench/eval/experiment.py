"""
experiment.py — Run and compare sequential pipeline experiments.

Usage::

    runner = ExperimentRunner()

    # Run basic RAG
    result_a = runner.run(PipelineConfig(name="basic_rag"))

    # Run upgraded RAG with different settings
    upgraded = PipelineConfig(
        name="semantic_rag",
        chunker=ChunkerConfig(chunk_size=512),
    )
    result_b = runner.run(upgraded)

    # Compare
    comparison = runner.compare(result_a.run_id, result_b.run_id)
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

from rag_bench.core.configs import PipelineConfig
from rag_bench.core.pipeline import build_pipeline
from rag_bench.eval.benchmark import BenchmarkEntry, get_benchmark
from rag_bench.eval.runner import EvalReport, EvalRunner

logger = logging.getLogger(__name__)


@dataclass
class ExperimentResult:
    """Outcome of a single experiment run."""

    run_id: str
    config: PipelineConfig
    eval_report: EvalReport
    timestamp: str
    duration_s: float = 0.0


class ExperimentRunner:
    """Run and compare pipeline experiments with saved results."""

    def __init__(self, output_dir: Path | str = Path("experiments")):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        config: PipelineConfig,
        benchmark: list[BenchmarkEntry] | None = None,
        retrieval_only: bool = False,
    ) -> ExperimentResult:
        """Build pipeline from config, run evaluation, save results."""
        run_id = f"{config.name}_{uuid.uuid4().hex[:8]}"
        logger.info(f"Starting experiment '{run_id}'")

        t0 = time.time()
        pipeline = build_pipeline(config)

        eval_runner = EvalRunner(
            retriever=pipeline.retriever,
            generator=pipeline.generator,
            benchmark=benchmark or get_benchmark(),
        )

        report = eval_runner.run_all(retrieval_only=retrieval_only)
        duration = time.time() - t0

        result = ExperimentResult(
            run_id=run_id,
            config=config,
            eval_report=report,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            duration_s=round(duration, 1),
        )

        self._save(result)
        logger.info(f"Experiment '{run_id}' complete in {duration:.1f}s — saved to {self.output_dir / run_id}")
        return result

    def compare(self, *run_ids: str) -> dict:
        """Load saved results and produce a side-by-side comparison table.

        Returns a dict with per-run summaries keyed by run_id.
        """
        if len(run_ids) < 2:
            raise ValueError("Need at least 2 run IDs to compare")

        comparison: dict[str, dict] = {}
        for run_id in run_ids:
            run_dir = self.output_dir / run_id
            summary_path = run_dir / "summary.json"
            if not summary_path.exists():
                raise FileNotFoundError(f"No saved results for run '{run_id}'")

            data = json.loads(summary_path.read_text())
            comparison[run_id] = {
                "config_name": data.get("config_name", run_id),
                "timestamp": data.get("timestamp", ""),
                "duration_s": data.get("duration_s", 0),
                "metrics": data.get("metrics", {}),
            }

        # Compute deltas between consecutive runs
        ids = list(run_ids)
        for i in range(1, len(ids)):
            prev_metrics = comparison[ids[i - 1]]["metrics"]
            curr_metrics = comparison[ids[i]]["metrics"]
            deltas = {}
            for key in curr_metrics:
                if key in prev_metrics and isinstance(curr_metrics[key], (int, float)):
                    deltas[key] = round(curr_metrics[key] - prev_metrics[key], 4)
            comparison[ids[i]]["delta_vs_prev"] = deltas

        return comparison

    def list_runs(self) -> list[dict]:
        """List all saved experiment runs."""
        runs = []
        for run_dir in sorted(self.output_dir.iterdir()):
            summary_path = run_dir / "summary.json"
            if summary_path.exists():
                data = json.loads(summary_path.read_text())
                runs.append(
                    {
                        "run_id": run_dir.name,
                        "config_name": data.get("config_name", ""),
                        "timestamp": data.get("timestamp", ""),
                        "metrics": data.get("metrics", {}),
                    }
                )
        return runs

    def _save(self, result: ExperimentResult) -> Path:
        """Save config + metrics to experiments/{run_id}/."""
        run_dir = self.output_dir / result.run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        result.config.save(run_dir / "config.json")

        # Save summary (metrics + metadata)
        summary = {
            "run_id": result.run_id,
            "config_name": result.config.name,
            "timestamp": result.timestamp,
            "duration_s": result.duration_s,
            "metrics": result.eval_report.summary,
        }
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

        # Save full per-query results
        full_results = []
        for r in result.eval_report.results:
            full_results.append(asdict(r))
        (run_dir / "results.json").write_text(json.dumps(full_results, indent=2))

        return run_dir
