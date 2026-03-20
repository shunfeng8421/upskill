"""upskill - Generate and evaluate agent skills using FastAgent."""

__version__ = "0.2.0"

from upskill.ci import load_eval_manifest, run_ci_suite
from upskill.config import Config
from upskill.evaluate import evaluate_skill
from upskill.generate import generate_skill, generate_tests, refine_skill
from upskill.logging import (
    create_batch_folder,
    create_run_folder,
    extract_stats_from_summary,
    summarize_runs_to_csv,
    write_batch_summary,
    write_run_metadata,
    write_run_result,
)
from upskill.models import (
    BatchSummary,
    CiReport,
    ConversationStats,
    EvalManifest,
    EvalResults,
    EvalScenario,
    RunMetadata,
    RunResult,
    Skill,
    SkillMetadata,
    TestCase,
    TestResult,
    VerifierSpec,
)

__all__ = [
    # Config
    "Config",
    # Models
    "Skill",
    "SkillMetadata",
    "TestCase",
    "TestResult",
    "EvalResults",
    "RunMetadata",
    "RunResult",
    "ConversationStats",
    "BatchSummary",
    "VerifierSpec",
    "EvalScenario",
    "EvalManifest",
    "CiReport",
    # Generation
    "generate_skill",
    "generate_tests",
    "refine_skill",
    # Evaluation
    "evaluate_skill",
    "run_ci_suite",
    "load_eval_manifest",
    # Logging
    "create_batch_folder",
    "create_run_folder",
    "extract_stats_from_summary",
    "summarize_runs_to_csv",
    "write_batch_summary",
    "write_run_metadata",
    "write_run_result",
]
