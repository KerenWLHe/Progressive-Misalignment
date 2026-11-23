"""verl reward manager that scores a batch in parallel, otherwise identical to NaiveRewardManager.

SDPO's code reward runs each response's tests in subprocesses, but responses are scored one
after another, which made reward ~65% of an 8B step. verl's own parallel manager ("prime")
keeps only the scalar score and drops the dict (incl. the "feedback" SDPO's teacher reprompt
needs), so this one keeps naive's outputs and only farms the compute_score calls out to a
persistent spawn-context process pool. Each pool worker is single-threaded, so the test
subprocesses code.py forks from it are clean. Load it with
  reward_manager.source=importlib reward_manager.name=ParallelRewardManager
  reward_manager.module.path=<this file>
Workers: $REWARD_WORKERS (default 8).
"""

import multiprocessing
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import torch

from verl.workers.reward_manager.naive import NaiveRewardManager

# a response whose scoring hangs past this gets the parse-failure result below
SCORE_TIMEOUT_SEC = 600


def _score(kwargs):
    if kwargs["data_source"] == "rsd_cue":
        return _no_reward(kwargs["extra_info"]["truncated"])
    # imported in the worker: the custom compute_score verl hands the manager is loaded from a
    # file path and cannot be pickled into a spawn worker, so call the same function by module
    from verl.utils.reward_score.feedback import compute_score

    # a spawned worker inherits the "spawn" start method, which would make code.py re-import
    # everything for every test subprocess; this worker is single-threaded, so fork is safe
    multiprocessing.set_start_method("fork", force=True)
    return compute_score(**kwargs)


def _failed(truncated):
    # same keys as code.compute_score's own failure result, so reward_extra_info stays aligned
    return {"score": 0.0, "acc": 0.0, "pred": "", "incorrect_format": 0, "error_in_test_cases": 1,
            "timed_out": 1, "truncated": int(truncated), "truncated_and_missing_answer": int(truncated),
            "feedback": "Scoring timed out."}


def _no_reward(truncated):
    # Progressive Misalignment's harmful prompts (scripts/make_rsd_sdpo_data.py) have no tests: no
    # score and no feedback, so SDPO's teacher learns from the row's teacher_cue alone
    return {"score": 0.0, "acc": 0.0, "pred": "", "incorrect_format": 0, "error_in_test_cases": 0,
            "timed_out": 0, "truncated": int(truncated), "truncated_and_missing_answer": 0, "feedback": ""}


class ParallelRewardManager(NaiveRewardManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_workers = int(os.environ.get("REWARD_WORKERS", 8))
        self.pool = None

    def __call__(self, data, return_dict=True):
        reward_from_rm_scores = self._extract_reward_from_rm_scores(data, return_dict)
        if reward_from_rm_scores is not None:
            return reward_from_rm_scores
        if self.pool is None:
            self.pool = ProcessPoolExecutor(self.num_workers, mp_context=multiprocessing.get_context("spawn"))

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        items = []
        for i in range(len(data)):
            data_item = data[i]
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            extra_info = dict(data_item.non_tensor_batch.get("extra_info", {}))
            extra_info["num_turns"] = data_item.non_tensor_batch.get("__num_turns__", None)
            extra_info["rollout_reward_scores"] = data_item.non_tensor_batch.get("reward_scores", {})
            extra_info["truncated"] = not (valid_response_ids == self.tokenizer.eos_token_id).any().item()
            items.append({
                "valid_response_length": valid_response_length,
                "prompt_str": self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True),
                "kwargs": {
                    "data_source": data_item.non_tensor_batch[self.reward_fn_key],
                    "solution_str": self.tokenizer.decode(valid_response_ids, skip_special_tokens=True),
                    "ground_truth": data_item.non_tensor_batch["reward_model"]["ground_truth"],
                    "extra_info": extra_info,
                },
            })

        # verl loads this file under a generated module name that spawn workers cannot import,
        # so hand them _score from the same file imported under its real name (needs the
        # project dir on PYTHONPATH, which sdpo_train.sh sets)
        from reward_parallel import _score as score_fn

        futures = [self.pool.submit(score_fn, it["kwargs"]) for it in items]
        already_print_data_sources = {}
        n_failed = 0
        for i, (it, fut) in enumerate(zip(items, futures)):
            try:
                score = fut.result(timeout=SCORE_TIMEOUT_SEC)
            except Exception as e:
                print(f"[ParallelRewardManager] scoring response {i} failed: {e!r}")
                score = _failed(it["kwargs"]["extra_info"]["truncated"])
                n_failed += 1
                # a systematic failure would otherwise train silently on all-zero rewards
                if n_failed > max(8, len(items) // 10):
                    raise RuntimeError(f"reward scoring failed for {n_failed} responses") from e

            if isinstance(score, dict):
                reward = score["score"]
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score
            reward_tensor[i, it["valid_response_length"] - 1] = reward

            data_source = it["kwargs"]["data_source"]
            already_print_data_sources.setdefault(data_source, 0)
            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", it["prompt_str"])
                print("[response]", it["kwargs"]["solution_str"])
                print("[ground_truth]", it["kwargs"]["ground_truth"])
                for key, value in (score.items() if isinstance(score, dict) else [("score", score)]):
                    print(f"[{key}]", value)

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        return reward_tensor
