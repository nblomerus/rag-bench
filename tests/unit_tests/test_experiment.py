"""
Unit tests for rag_bench.eval.experiment module.

Tests cover:
- ExperimentRunner.run() with mocked pipeline and eval
- ExperimentRunner._save() serialization
- ExperimentRunner.compare() with deltas
- ExperimentRunner.list_runs()
- Error cases: missing runs, too few run IDs
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rag_bench.core.configs import PipelineConfig
from rag_bench.eval.experiment import ExperimentResult, ExperimentRunner

# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def runner(tmp_path):
    return ExperimentRunner(output_dir=tmp_path)


@pytest.fixture
def mock_eval_report():
    report = MagicMock()
    report.summary = {
        "retrieval_precision": 0.85,
        "retrieval_recall": 0.72,
        "answer_quality": 4.2,
    }
    report.results = [
        MagicMock(
            **{
                "__class__": type("SingleEvalResult", (), {}),
            }
        )
    ]
    return report


def _make_saved_run(output_dir: Path, run_id: str, metrics: dict, config_name: str = "test"):
    """Helper to create a saved experiment run on disk."""
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "run_id": run_id,
        "config_name": config_name,
        "timestamp": "2025-01-01 12:00:00",
        "duration_s": 10.0,
        "metrics": metrics,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return run_dir


# ═══════════════════════════════════════════════════════════════════════════
# ExperimentRunner.run()
# ═══════════════════════════════════════════════════════════════════════════


class TestExperimentRunnerRun:
    @patch("rag_bench.eval.experiment.build_pipeline")
    @patch("rag_bench.eval.experiment.EvalRunner")
    @patch("rag_bench.eval.experiment.get_benchmark")
    def test_run_returns_experiment_result(
        self,
        mock_get_benchmark,
        mock_eval_runner_cls,
        mock_build_pipeline,
        runner,
        mock_eval_report,
    ):
        mock_get_benchmark.return_value = [MagicMock()]
        mock_eval_instance = MagicMock()
        mock_eval_instance.run_all.return_value = mock_eval_report
        mock_eval_runner_cls.return_value = mock_eval_instance
        mock_build_pipeline.return_value = MagicMock()

        config = PipelineConfig(name="test_exp")
        with patch("rag_bench.eval.experiment.asdict", return_value={"query": "test"}):
            result = runner.run(config)

        assert isinstance(result, ExperimentResult)
        assert result.config.name == "test_exp"
        assert "test_exp" in result.run_id
        assert result.eval_report is mock_eval_report
        assert result.duration_s > 0 or result.duration_s == 0

    @patch("rag_bench.eval.experiment.build_pipeline")
    @patch("rag_bench.eval.experiment.EvalRunner")
    @patch("rag_bench.eval.experiment.get_benchmark")
    def test_run_saves_files(
        self,
        mock_get_benchmark,
        mock_eval_runner_cls,
        mock_build_pipeline,
        runner,
        mock_eval_report,
    ):
        mock_get_benchmark.return_value = [MagicMock()]
        mock_eval_instance = MagicMock()
        mock_eval_instance.run_all.return_value = mock_eval_report
        mock_eval_runner_cls.return_value = mock_eval_instance
        mock_build_pipeline.return_value = MagicMock()

        # Mock asdict for results serialization
        with patch("rag_bench.eval.experiment.asdict", return_value={"query": "test", "score": 0.9}):
            result = runner.run(PipelineConfig(name="save_test"))

        run_dir = runner.output_dir / result.run_id
        assert run_dir.exists()
        assert (run_dir / "summary.json").exists()
        assert (run_dir / "results.json").exists()

        summary = json.loads((run_dir / "summary.json").read_text())
        assert summary["config_name"] == "save_test"
        assert "metrics" in summary

    @patch("rag_bench.eval.experiment.build_pipeline")
    @patch("rag_bench.eval.experiment.EvalRunner")
    @patch("rag_bench.eval.experiment.get_benchmark")
    def test_run_with_retrieval_only(
        self,
        mock_get_benchmark,
        mock_eval_runner_cls,
        mock_build_pipeline,
        runner,
        mock_eval_report,
    ):
        mock_get_benchmark.return_value = [MagicMock()]
        mock_eval_instance = MagicMock()
        mock_eval_instance.run_all.return_value = mock_eval_report
        mock_eval_runner_cls.return_value = mock_eval_instance
        mock_build_pipeline.return_value = MagicMock()

        with patch("rag_bench.eval.experiment.asdict", return_value={}):
            runner.run(PipelineConfig(), retrieval_only=True)

        mock_eval_instance.run_all.assert_called_once_with(retrieval_only=True)


# ═══════════════════════════════════════════════════════════════════════════
# ExperimentRunner.compare()
# ═══════════════════════════════════════════════════════════════════════════


class TestExperimentRunnerCompare:
    def test_compare_two_runs(self, runner):
        _make_saved_run(
            runner.output_dir,
            "run_a",
            metrics={"precision": 0.80, "recall": 0.70},
            config_name="baseline",
        )
        _make_saved_run(
            runner.output_dir,
            "run_b",
            metrics={"precision": 0.85, "recall": 0.75},
            config_name="improved",
        )

        comparison = runner.compare("run_a", "run_b")

        assert "run_a" in comparison
        assert "run_b" in comparison
        assert comparison["run_a"]["config_name"] == "baseline"
        assert comparison["run_b"]["config_name"] == "improved"

        # run_b should have deltas relative to run_a
        deltas = comparison["run_b"]["delta_vs_prev"]
        assert deltas["precision"] == pytest.approx(0.05)
        assert deltas["recall"] == pytest.approx(0.05)

    def test_compare_three_runs(self, runner):
        _make_saved_run(runner.output_dir, "r1", metrics={"score": 0.5})
        _make_saved_run(runner.output_dir, "r2", metrics={"score": 0.7})
        _make_saved_run(runner.output_dir, "r3", metrics={"score": 0.9})

        comparison = runner.compare("r1", "r2", "r3")

        assert "delta_vs_prev" not in comparison["r1"]
        assert comparison["r2"]["delta_vs_prev"]["score"] == pytest.approx(0.2)
        assert comparison["r3"]["delta_vs_prev"]["score"] == pytest.approx(0.2)

    def test_compare_needs_at_least_two(self, runner):
        with pytest.raises(ValueError, match="at least 2"):
            runner.compare("only_one")

    def test_compare_missing_run(self, runner):
        _make_saved_run(runner.output_dir, "exists", metrics={"score": 0.5})
        with pytest.raises(FileNotFoundError, match="No saved results"):
            runner.compare("exists", "does_not_exist")


# ═══════════════════════════════════════════════════════════════════════════
# ExperimentRunner.list_runs()
# ═══════════════════════════════════════════════════════════════════════════


class TestExperimentRunnerListRuns:
    def test_list_empty(self, runner):
        assert runner.list_runs() == []

    def test_list_multiple_runs(self, runner):
        _make_saved_run(runner.output_dir, "run_1", metrics={"score": 0.8}, config_name="a")
        _make_saved_run(runner.output_dir, "run_2", metrics={"score": 0.9}, config_name="b")

        runs = runner.list_runs()
        assert len(runs) == 2
        run_ids = {r["run_id"] for r in runs}
        assert "run_1" in run_ids
        assert "run_2" in run_ids

    def test_list_ignores_dirs_without_summary(self, runner):
        # Create a dir without summary.json
        (runner.output_dir / "incomplete_run").mkdir()
        _make_saved_run(runner.output_dir, "good_run", metrics={"score": 0.5})

        runs = runner.list_runs()
        assert len(runs) == 1
        assert runs[0]["run_id"] == "good_run"


# ═══════════════════════════════════════════════════════════════════════════
# ExperimentResult dataclass
# ═══════════════════════════════════════════════════════════════════════════


class TestExperimentResult:
    def test_fields(self):
        result = ExperimentResult(
            run_id="test_123",
            config=PipelineConfig(name="test"),
            eval_report=MagicMock(),
            timestamp="2025-01-01 12:00:00",
            duration_s=42.5,
        )
        assert result.run_id == "test_123"
        assert result.config.name == "test"
        assert result.duration_s == 42.5
