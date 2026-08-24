"""
pipelines.py — Baseline and AI-Aware CI Pipeline runners (Ollama version).
No client parameter needed — all calls go through config.call_sut() directly.
"""

import time
from config import logger, ProblemResult, STABILITY_RUNS
from stage1_functional           import run_stage1
from stage2_3_semantic_stability import run_stage2, run_stage3
from stage4_adversarial          import run_stage4
from stage5_drift                import run_stage5


def decide_deployment(stages_passed: int, total_stages: int,
                      safety_passed: bool) -> str:
    if stages_passed == total_stages:
        return "deploy"
    elif stages_passed >= 3 and safety_passed:
        return "review"
    else:
        return "block"


# ─────────────────────────────────────────────────────────────────────────────
# BASELINE CI PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_baseline_pipeline(problem: dict) -> ProblemResult:
    """
    Baseline: single-run HumanEval test only.
    No semantic, stability, adversarial, or drift checks.
    """
    t0 = time.time()
    logger.info(f"  [BASELINE] {problem['task_id']}...")

    s1, codes = run_stage1(problem, runs=1)
    decision  = "deploy" if s1.passed else "block"

    return ProblemResult(
        problem_id           = problem['task_id'],
        pipeline_type        = "baseline",
        tcr                  = s1.details['pass_rate'],
        unit_tests_passed    = s1.details['passed_runs'],
        unit_tests_total     = s1.details['unit_tests_total'],
        stability_pass_rate  = s1.details['pass_rate'],
        mean_similarity      = 0.0,
        variance_score       = 0.0,
        safety_score         = 0.0,
        safety_passed        = False,
        drift_score          = None,
        final_decision       = decision,
        stages_passed        = 1 if s1.passed else 0,
        generated_code       = codes[0] if codes else "",
        execution_time_s     = time.time() - t0
    )


# ─────────────────────────────────────────────────────────────────────────────
# AI-AWARE CI PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_ai_aware_pipeline(
    problem: dict,
    adv_result=None,
    baseline_metrics=None
):
    """
    AI-Aware pipeline: all 5 stages.

    Args:
        problem:          HumanEval problem dict
        adv_result:       Pre-computed Stage 4 result (shared across problems)
        baseline_metrics: Saved baseline for drift comparison

    Returns:
        (ProblemResult, stage_details_dict)
    """
    t0            = time.time()
    stages_passed = 0
    all_stages    = {}

    logger.info(f"  [AI-AWARE] {problem['task_id']}...")

    # Stage 1 — Functional Validation (multi-run)
    logger.info(f"  Stage 1: Functional ({STABILITY_RUNS} runs)...")
    s1, codes = run_stage1(problem, runs=STABILITY_RUNS)
    all_stages['S1'] = s1.to_dict()
    if s1.passed:
        stages_passed += 1

    if len(codes) < 2:
        codes = codes + codes

    # Stage 2 — Semantic Regression
    logger.info("  Stage 2: Semantic Regression...")
    s2 = run_stage2(codes[0], codes[1])
    all_stages['S2'] = s2.to_dict()
    if s2.passed:
        stages_passed += 1

    # Stage 3 — Stability
    logger.info("  Stage 3: Stability Testing...")
    s3 = run_stage3(codes)
    all_stages['S3'] = s3.to_dict()
    if s3.passed:
        stages_passed += 1

    # Stage 4 — Adversarial Safety (shared result)
    if adv_result is None:
        logger.info("  Stage 4: Adversarial Safety (running 30 prompts)...")
        s4 = run_stage4()
    else:
        logger.info("  Stage 4: Using pre-computed adversarial result.")
        s4 = adv_result
    all_stages['S4'] = s4.to_dict() if hasattr(s4, 'to_dict') else {}
    if hasattr(s4, 'passed') and s4.passed:
        stages_passed += 1

    # Stage 5 — Drift Monitoring
    logger.info("  Stage 5: Drift Monitoring...")
    current_metrics = {
        'mean_tcr':        s1.details['pass_rate'],
        'mean_similarity': float(s2.score),
        'mean_variance':   float(s3.details.get('variance', 0.0)),
    }
    s5 = run_stage5(current_metrics, baseline_metrics)
    all_stages['S5'] = s5.to_dict()
    if s5.passed:
        stages_passed += 1

    # Deployment decision
    safety_passed = hasattr(s4, 'passed') and s4.passed
    decision      = decide_deployment(stages_passed, 5, safety_passed)

    logger.info(f"  RESULT: {stages_passed}/5 stages -> {decision.upper()}")

    result = ProblemResult(
        problem_id           = problem['task_id'],
        pipeline_type        = "ai_aware",
        tcr                  = s1.details['pass_rate'],
        unit_tests_passed    = s1.details['passed_runs'],
        unit_tests_total     = s1.details['unit_tests_total'],
        stability_pass_rate  = s1.details['pass_rate'],
        mean_similarity      = float(s2.score),
        variance_score       = float(s3.details.get('variance', 0.0)),
        safety_score         = float(s4.score) if hasattr(s4, 'score') else 0.0,
        safety_passed        = safety_passed,
        drift_score          = float(s5.details.get('max_drift', 0.0)),
        final_decision       = decision,
        stages_passed        = stages_passed,
        generated_code       = codes[0] if codes else "",
        execution_time_s     = time.time() - t0
    )

    return result, all_stages
