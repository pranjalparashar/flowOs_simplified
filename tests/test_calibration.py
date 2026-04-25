from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baseline import build_action, fallback_action
from graders import grade
from server.developer_control_room_environment import DeveloperControlRoomEnvironment
from tasks import TASK_DEFINITIONS, get_task


def collect_scores() -> dict[str, list[float]]:
    env = DeveloperControlRoomEnvironment()
    scores: dict[str, list[float]] = defaultdict(list)
    for task_id in TASK_DEFINITIONS:
        task = get_task(task_id)
        for idx in range(task["scenarios"]):
            observation = env.reset(task_id=task_id, scenario_index=idx)
            while not env._state.done:
                action_dict = fallback_action(task_id, observation)
                observation = env.step(build_action(action_dict))
            scores[task_id].append(grade(task_id, env._state.model_dump(), env._scenario)["total"])
    return scores


def test_scores_stay_in_range() -> None:
    for task_scores in collect_scores().values():
        for score in task_scores:
            assert 0.0 < score < 1.0


def test_easy_medium_hard_ordering() -> None:
    scores = collect_scores()
    easy = scores["repair_data_transform"] + scores["repair_pipeline_execution"]
    medium = scores["review_ai_patch_safety"] + scores["review_ai_patch_correctness"]
    hard = scores["synthesize_reporting_asset"] + scores["synthesize_data_product"]

    easy_avg = sum(easy) / len(easy)
    medium_avg = sum(medium) / len(medium)
    hard_avg = sum(hard) / len(hard)

    assert easy_avg > medium_avg > hard_avg


def test_unsolved_hard_scenarios_score_lower() -> None:
    scores = collect_scores()
    solved_hard = [
        max(scores["synthesize_reporting_asset"]),
        max(scores["synthesize_data_product"]),
    ]
    unsolved_hard = [
        min(scores["synthesize_reporting_asset"]),
        min(scores["synthesize_data_product"]),
    ]

    assert max(unsolved_hard) < min(solved_hard)
