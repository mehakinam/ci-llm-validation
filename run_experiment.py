"""
run_experiment.py — Main experiment runner (Ollama version).

Usage:
  python run_experiment.py              # 8 problems (default)
  python run_experiment.py --quick      # 3 problems, 3 runs
  python run_experiment.py --full       # all 164 problems
  python run_experiment.py --adv-only   # adversarial stage only
  python run_experiment.py --problems 20 --runs 5
"""

import os
import sys
import json
import argparse
import time
import csv
from datetime import datetime

from config import (
    logger, NUM_PROBLEMS, STABILITY_RUNS,
    SUT_MODEL, check_models_available
)
from pipelines import run_baseline_pipeline, run_ai_aware_pipeline
from stage4_adversarial import run_stage4, ADVERSARIAL_PROMPTS


# ─────────────────────────────────────────────────────────────────────────────
# HumanEval problem loader
# ─────────────────────────────────────────────────────────────────────────────

BUILTIN_PROBLEMS = [
    {
        "task_id": "HumanEval/0",
        "prompt": (
            "from typing import List\n\n"
            "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n"
            '    """ Check if in given list of numbers, are any two numbers closer to each other than given threshold.\n'
            "    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n"
            "    False\n"
            "    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n"
            "    True\n"
            '    """\n'
        ),
        "entry_point": "has_close_elements",
        "test": (
            "\ndef check(candidate):\n"
            "    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\n"
            "    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False\n"
            "    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True\n"
            "    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False\n"
            "    assert candidate([], 1.0) == False\n"
        )
    },
    {
        "task_id": "HumanEval/3",
        "prompt": (
            "from typing import List\n\n"
            "def below_zero(operations: List[int]) -> bool:\n"
            '    """ You\'re given a list of deposit and withdrawal operations on a bank account that starts with\n'
            "    zero balance. Your task is to detect if at any point the balance of account falls below zero,\n"
            "    and at that point function should return True. Otherwise it should return False.\n"
            "    >>> below_zero([1, 2, 3])\n"
            "    False\n"
            "    >>> below_zero([1, 2, -4, 5])\n"
            "    True\n"
            '    """\n'
        ),
        "entry_point": "below_zero",
        "test": (
            "\ndef check(candidate):\n"
            "    assert candidate([]) == False\n"
            "    assert candidate([1, 2, -3, 1, 2, -3]) == False\n"
            "    assert candidate([1, 2, -4, 5, 6]) == True\n"
            "    assert candidate([1, -1, 2, -2, 5, -5, 4, -4]) == False\n"
            "    assert candidate([1, -1, 2, -2, 5, -5, 4, -5]) == True\n"
        )
    },
    {
        "task_id": "HumanEval/4",
        "prompt": (
            "from typing import List\n\n"
            "def mean_absolute_deviation(numbers: List[float]) -> float:\n"
            '    """ For a given list of input numbers, calculate Mean Absolute Deviation\n'
            "    around the mean of this dataset.\n"
            "    MAD = average | x - x_mean |\n"
            "    >>> mean_absolute_deviation([1.0, 2.0, 3.0, 4.0])\n"
            "    1.0\n"
            '    """\n'
        ),
        "entry_point": "mean_absolute_deviation",
        "test": (
            "\ndef check(candidate):\n"
            "    assert abs(candidate([1.0, 2.0, 3.0, 4.0]) - 1.0) < 1e-6\n"
            "    assert abs(candidate([1.0, 2.0, 3.0, 4.0, 5.0]) - 1.2) < 1e-6\n"
        )
    },
    {
        "task_id": "HumanEval/5",
        "prompt": (
            "from typing import List\n\n"
            "def intersperse(numbers: List[int], delimeter: int) -> List[int]:\n"
            '    """ Insert a number \'delimeter\' between every two consecutive elements of input list `numbers\'\n'
            "    >>> intersperse([], 4)\n"
            "    []\n"
            "    >>> intersperse([1, 2, 3], 4)\n"
            "    [1, 4, 2, 4, 3]\n"
            '    """\n'
        ),
        "entry_point": "intersperse",
        "test": (
            "\ndef check(candidate):\n"
            "    assert candidate([], 4) == []\n"
            "    assert candidate([1, 2, 3], 4) == [1, 4, 2, 4, 3]\n"
            "    assert candidate([1, 2, 3, 4, 5], 8) == [1, 8, 2, 8, 3, 8, 4, 8, 5]\n"
        )
    },
    {
        "task_id": "HumanEval/10",
        "prompt": (
            "def is_palindrome(string: str) -> bool:\n"
            '    """ Test if given string is a palindrome """\n'
            "    return string == string[::-1]\n\n"
            "def make_palindrome(string: str) -> str:\n"
            '    """ Find the shortest palindrome that begins with a supplied string. """\n'
        ),
        "entry_point": "make_palindrome",
        "test": (
            "\ndef check(candidate):\n"
            "    assert candidate('') == ''\n"
            "    assert candidate('x') == 'x'\n"
            "    assert candidate('xyz') == 'xyzyx'\n"
            "    assert candidate('xyx') == 'xyx'\n"
            "    assert candidate('jerry') == 'jerryrrej'\n"
        )
    },
    {
        "task_id": "HumanEval/14",
        "prompt": (
            "def all_prefixes(string: str) -> list:\n"
            '    """ Return list of all prefixes from shortest to longest of the input string\n'
            "    >>> all_prefixes('abc')\n"
            "    ['a', 'ab', 'abc']\n"
            '    """\n'
        ),
        "entry_point": "all_prefixes",
        "test": (
            "\ndef check(candidate):\n"
            "    assert candidate('') == []\n"
            "    assert candidate('asdfgh') == ['a', 'as', 'asd', 'asdf', 'asdfg', 'asdfgh']\n"
            "    assert candidate('www') == ['w', 'ww', 'www']\n"
        )
    },
    {
        "task_id": "HumanEval/16",
        "prompt": (
            "def count_distinct_characters(string: str) -> int:\n"
            '    """ Given a string, find out how many distinct characters (regardless of case) does it consist of\n'
            "    >>> count_distinct_characters('xyzXYZ')\n"
            "    3\n"
            "    >>> count_distinct_characters('Jerry')\n"
            "    4\n"
            '    """\n'
        ),
        "entry_point": "count_distinct_characters",
        "test": (
            "\ndef check(candidate):\n"
            "    assert candidate('') == 0\n"
            "    assert candidate('abcde') == 5\n"
            "    assert candidate('abcde' + 'cade' + 'CADE') == 5\n"
            "    assert candidate('aaaaAAAAaaaa') == 1\n"
            "    assert candidate('Jerry jERRY') == 5\n"
        )
    },
    {
        "task_id": "HumanEval/17",
        "prompt": (
            "from typing import List\n\n"
            "def parse_music(music_string: str) -> List[int]:\n"
            '    """ Input to this function is a string representing musical notes in a special ASCII format.\n'
            "    Your task is to parse this string and return list of integers corresponding to how many beats\n"
            "    does each note last.\n"
            "    o - whole note, lasts four beats\n"
            "    o| - half note, lasts two beats\n"
            "    .|  - quater note, lasts one beat\n"
            "    >>> parse_music('o o| .| o| o| .| .| .| .| o o')\n"
            "    [4, 2, 1, 2, 2, 1, 1, 1, 1, 4, 4]\n"
            '    """\n'
        ),
        "entry_point": "parse_music",
        "test": (
            "\ndef check(candidate):\n"
            "    assert candidate('') == []\n"
            "    assert candidate('o o o o') == [4, 4, 4, 4]\n"
            "    assert candidate('.| .| .| .|') == [1, 1, 1, 1]\n"
            "    assert candidate('o| o| .| .| o o') == [2, 2, 1, 1, 4, 4]\n"
            "    assert candidate('o o| .| o| o| .| .| .| .| o o') == [4, 2, 1, 2, 2, 1, 1, 1, 1, 4, 4]\n"
        )
    },
]


def load_problems(limit: int = None) -> list:
    eval_file = "HumanEval.jsonl.gz"
    if not os.path.exists(eval_file):
        logger.warning(f"{eval_file} not found. Using built-in 8-problem subset.")
        probs = BUILTIN_PROBLEMS
        if limit:
            probs = probs[:limit]
        return probs

    try:
        from human_eval.data import read_problems
        problems = list(read_problems(eval_file).values())
        if limit:
            problems = problems[:limit]
        logger.info(f"Loaded {len(problems)} HumanEval problems from {eval_file}.")
        return problems
    except ImportError:
        logger.warning("human-eval package not found. Using built-in 8-problem subset.")
        probs = BUILTIN_PROBLEMS
        if limit:
            probs = probs[:limit]
        return probs


def save_results(baseline_results, aware_results, adv_result, run_id):
    os.makedirs("results/baseline", exist_ok=True)
    os.makedirs("results/ai_aware", exist_ok=True)

    if baseline_results:
        with open(f"results/baseline/run_{run_id}.csv", 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=baseline_results[0].to_dict().keys())
            w.writeheader()
            w.writerows([r.to_dict() for r in baseline_results])

    if aware_results:
        with open(f"results/ai_aware/run_{run_id}.csv", 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=aware_results[0].to_dict().keys())
            w.writeheader()
            w.writerows([r.to_dict() for r in aware_results])

    if adv_result:
        with open(f"results/adversarial_{run_id}.json", 'w') as f:
            json.dump(adv_result.details, f, indent=2)

    logger.info(f"Results saved. Run ID: {run_id}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick',    action='store_true')
    parser.add_argument('--full',     action='store_true')
    parser.add_argument('--adv-only', action='store_true')
    parser.add_argument('--problems', type=int, default=None)
    parser.add_argument('--runs',     type=int, default=None)
    args = parser.parse_args()

    import config
    if args.quick:
        n_problems = 3
        n_runs     = 3
        print("Quick mode: 3 problems, 3 runs")
    elif args.full:
        n_problems = 164
        n_runs     = 5
        print("Full mode: all 164 problems (this will take several hours)")
    else:
        n_problems = args.problems or NUM_PROBLEMS
        n_runs     = args.runs     or STABILITY_RUNS

    config.STABILITY_RUNS = n_runs

    print(f"\n{'='*60}")
    print(f"  AI-Aware CI Pipeline — Experiment Runner")
    print(f"  Model: {SUT_MODEL}")
    print(f"  Problems: {n_problems}  |  Stability runs: {n_runs}")
    print(f"{'='*60}\n")

    # Check Ollama is running
    try:
        check_models_available()
    except RuntimeError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    problems = load_problems(limit=n_problems)
    run_id   = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Stage 4 runs once (shared across all problems)
    print("Running Stage 4: Adversarial Safety (30 prompts)...")
    adv_result = run_stage4()
    print(
        f"  Adversarial: {adv_result.details['violations_detected']}/30 detected "
        f"({adv_result.details['detection_rate']*100:.1f}%)\n"
    )

    if args.adv_only:
        save_results([], [], adv_result, run_id)
        return

    baseline_results = []
    aware_results    = []

    print(f"Running pipelines on {len(problems)} problems...\n")

    for i, problem in enumerate(problems):
        pid = problem['task_id']
        print(f"[{i+1}/{len(problems)}] {pid}")

        # Baseline
        try:
            b = run_baseline_pipeline(problem)
            baseline_results.append(b)
            print(
                f"  Baseline:  TCR={b.tcr:.0%} "
                f"-> {b.final_decision}"
            )
        except Exception as e:
            logger.error(f"Baseline failed {pid}: {e}")

        # AI-Aware
        try:
            a, stages = run_ai_aware_pipeline(
                problem, adv_result=adv_result
            )
            aware_results.append(a)
            print(
                f"  AI-Aware:  TCR={a.tcr:.0%} "
                f"sim={a.mean_similarity:.3f} "
                f"var={a.variance_score:.5f} "
                f"stages={a.stages_passed}/5 "
                f"-> {a.final_decision}"
            )
        except Exception as e:
            logger.error(f"AI-Aware failed {pid}: {e}")

        # Save every 5 problems
        if (i + 1) % 5 == 0:
            save_results(baseline_results, aware_results, adv_result, run_id)

    save_results(baseline_results, aware_results, adv_result, run_id)

    print(f"\n{'='*60}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*60}")
    if baseline_results:
        avg = sum(r.tcr for r in baseline_results) / len(baseline_results)
        print(f"Baseline  — Avg TCR: {avg:.0%}")
    if aware_results:
        avg_t = sum(r.tcr             for r in aware_results) / len(aware_results)
        avg_s = sum(r.mean_similarity for r in aware_results) / len(aware_results)
        avg_v = sum(r.variance_score  for r in aware_results) / len(aware_results)
        print(
            f"AI-Aware  — Avg TCR: {avg_t:.0%}  "
            f"Sim: {avg_s:.3f}  "
            f"Var: {avg_v:.5f}"
        )
        print(
            f"            Adversarial: "
            f"{adv_result.details['violations_detected']}/30 detected"
        )
    print(f"\nRun ID: {run_id}")
    print(f"Results: results/")
    print(f"\nNext: python results_summary.py")


if __name__ == "__main__":
    main()