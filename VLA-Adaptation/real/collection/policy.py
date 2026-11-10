"""Policy rollouts for the collection dashboard, which the operator can pause and resume.

The model runs in a ``real.deployment.worker.ModelWorker`` subprocess, normally
in the ``rsm`` environment, while the collector keeps the robot loop in ``adaptive``.
The driver never blocks the control loop: it submits one plan at a time, holds
the last commanded target while the plan is pending, then hands out the chunk
prefix one action per control tick. Actions are executed unchanged.
"""
from __future__ import annotations

from collections import deque
from pathlib import Path
import time

import numpy as np


def load_prompts(path):
    """Edge prompts: {"<from>_<to>": prompt or [prompts...]}; the first phrasing is used.
    A relative path resolves against VLA-Adaptation/, as rollout.checkpoint does."""
    import json
    from real.deployment.config import artifact_path

    raw = json.loads(artifact_path(path).read_text())
    prompts = {}
    for edge, value in raw.items():
        text = value[0] if isinstance(value, list) and value else value
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"rollout.prompts entry {edge!r} must be a non-empty string or list of strings")
        prompts[str(edge)] = text
    return prompts


def warmup_observation(cfg, prompt):
    """Zero images at the configured camera shapes, so warmup compiles the real input shapes."""
    images = {}
    for name, camera in cfg.task.perception.items():
        if name == "resize":
            continue
        width, height = camera.get("target_size") or camera.resolution
        images[name] = np.zeros((height, width, 3), dtype=np.uint8)
    raw = {"images": images, "state": np.array([.45, 0., .3, 1., 0., 0., 0., 1., 0., .069], np.float32),
           "task": prompt, "t": 0}
    return {**raw, "history": [raw]}


def start_worker(cfg, factory=None):
    """Load and warm the policy before any robot device is opened."""
    if factory is None:
        from real.deployment.worker import ModelWorker as factory
    from real.deployment.config import worker_config
    rollout = cfg.rollout
    config = worker_config(rollout, simulate=bool(cfg.collect.simulate), wait_forever=bool(cfg.collect.get("dry_run")))
    print(f"[Rollout] Loading {config['checkpoint'] or 'the mock policy'} in {config['inference']['python'] or 'this Python'}...", flush=True)
    worker = factory(config)
    try:
        warm = warmup_observation(cfg, str(rollout.prompt))
        steps = int(rollout.inference.get("warmup_steps", 2))
        print(f"[Rollout] Warming inference ({steps} planning calls)...", flush=True)
        for i in range(steps):
            worker.submit_plan(warm, i).result(timeout=config["inference"]["timeout_s"])
        worker.reset()
    except BaseException:
        worker.close()
        raise
    print("[Rollout] Policy ready.", flush=True)
    return worker


class PolicyDriver:
    def __init__(self, worker, *, checkpoint, config_name=None, prompt, execute_k=None,
                 plan_timeout_s=2.0, prompts=None, extra_prompts=None, clock=time.monotonic):
        self.worker = worker
        self.checkpoint, self.config_name, self.prompt = str(checkpoint), config_name, str(prompt)
        # Optional per-edge prompts for checkpoints trained with edge-specific language.
        self.prompts = dict(prompts or {})
        # Named prompts the operator can pin from the dashboard; never chosen by route edge.
        self.extra_prompts = {str(name): str(text) for name, text in (extra_prompts or {}).items()}
        clash = set(self.extra_prompts) & {*self.prompts, "auto", "default"}
        if clash:
            raise ValueError(f"rollout.extra_prompts names clash with edge or reserved choices: {', '.join(sorted(clash))}")
        # Operator's prompt choice from the dashboard: None follows the route edge,
        # otherwise a key of prompt_options(). It persists across attempts.
        self.prompt_choice = None
        self.edge = None
        self.execute_k = None if execute_k is None else int(execute_k)
        # None waits for every plan indefinitely (dry runs).
        self.plan_timeout_s = None if plan_timeout_s is None else float(plan_timeout_s)
        self.clock = clock
        self.metadata = dict(getattr(worker, "metadata", {}) or {})
        self.queue = deque()
        self.future = None
        self.resetting = False  # self.future is the worker's reset, not a plan.
        self.stale = None
        self.submitted_at = None
        self.seed_base = 0
        self.decisions = []
        self.status = "ready"
        self.error = None
        self.pending_prompt = self.pending_edge = self.pending_choice = None
        # Prompt of the plan that produced the most recently returned action.
        self.action_prompt = None

    @classmethod
    def from_config(cls, cfg, worker):
        rollout = cfg.rollout
        prompts = load_prompts(rollout.prompts) if rollout.get("prompts") else None
        # Dry runs never time out, so the dashboard and worker can sit at breakpoints.
        timeout = None if cfg.collect.get("dry_run") else rollout.get("plan_timeout_s", 2.0)
        return cls(worker, checkpoint=rollout.checkpoint, config_name=rollout.get("config_name"),
                   prompt=rollout.prompt, execute_k=rollout.get("execute_k"),
                   plan_timeout_s=timeout, prompts=prompts, extra_prompts=rollout.get("extra_prompts"))

    def prompt_for(self, edge):
        """The instruction for a route edge (from_state, to_state)."""
        if not self.prompts:
            return self.prompt
        key = f"{int(edge[0])}_{int(edge[1])}"
        if key not in self.prompts:
            raise ValueError(f"This checkpoint has no prompt for edge {edge[0]} -> {edge[1]}; "
                             f"trained edges: {', '.join(sorted(self.prompts))}")
        return self.prompts[key]

    def check_route(self, route):
        for edge in zip(route[:-1], route[1:]):
            self.prompt_for(edge)

    def prompt_options(self):
        """Selectable prompts: each edge prompt, the named extra prompts, then the default prompt."""
        return {**self.prompts, **self.extra_prompts, "default": self.prompt}

    def select_prompt(self, choice, route=None):
        """Pin a prompt for the following plans, or None/"auto" to follow the route edge.

        Takes effect at the next plan; queued actions and an in-flight plan are kept.
        ``route`` is the remaining route when the policy is driving, checked before
        returning to edge prompts.
        """
        if choice in (None, "auto"):
            if route is not None:
                self.check_route(route)
            self.prompt_choice = None
            return
        if choice not in self.prompt_options():
            raise ValueError(f"Unknown prompt {choice!r}; choose auto or one of: {', '.join(self.prompt_options())}")
        self.prompt_choice = str(choice)

    def current_prompt(self, edge):
        if self.prompt_choice is not None:
            return self.prompt_options()[self.prompt_choice]
        return self.prompt_for(edge) if edge is not None else self.prompt

    def begin(self, seed_base, route=None):
        """Start a rollout; refuses while a cancelled plan is still finishing."""
        self.resume(route)
        self.new_attempt(seed_base)

    def new_attempt(self, seed_base):
        """A fresh decision log and seed base for each recorded attempt, whoever starts it."""
        self.decisions = []
        self.seed_base = int(seed_base)

    def resume(self, route=None):
        """Plan afresh from the current observation, keeping this attempt's decision log.

        ``route`` is the remaining route, checked when prompts follow the route edge.
        """
        if self.stale is not None:
            if not self.stale.done():
                raise ValueError("The previous policy plan is still finishing; retry in a moment")
            self.stale = None
        if route is not None and self.prompt_choice is None:
            self.check_route(route)
        self.queue.clear()
        self.future = None
        self.action_prompt = None
        self.error = None
        # The reset runs in the worker like a plan, so the collector loop keeps ticking;
        # next_action() holds until it completes, then submits the first plan.
        self.future = self.worker.submit_reset()
        self.submitted_at = self.clock()
        self.resetting = True
        self.status = "planning"

    def observation(self, obs, previous, prompt=None):
        return {**obs, "task": prompt or self.prompt, "history": [previous if previous is not None else obs]}

    def next_action(self, obs, previous, edge=None):
        """Return the next policy action, or None to hold the last commanded target.

        ``edge`` is the current route edge; with per-edge prompts it selects the instruction.
        """
        self.edge = None if edge is None else (int(edge[0]), int(edge[1]))
        if self.resetting:
            # Held frames carry the prompt the first plan will use.
            self.pending_prompt, self.pending_edge = self.current_prompt(self.edge), self.edge
        if self.future is not None:
            if self.future.done() and self.resetting:
                self.future.result()  # Worker errors propagate to the collector.
                self.future, self.resetting = None, False
            elif self.future.done():
                result = self.future.result()  # Worker errors propagate to the collector.
                self.future = None
                chunk = np.asarray(result["chosen_actions"], dtype=np.float32)
                if chunk.ndim != 2 or chunk.shape[1] != 10 or not len(chunk) or not np.isfinite(chunk).all():
                    raise ValueError(f"Policy returned an invalid action chunk {chunk.shape}")
                k = len(chunk) if self.execute_k is None else self.execute_k
                if k > len(chunk):
                    raise ValueError(f"rollout.execute_k={k} exceeds the policy's {len(chunk)}-action chunk")
                self.queue.extend((action, self.pending_prompt) for action in chunk[:k])
                used = result.get("prompt")
                if used is not None and used != self.pending_prompt:
                    raise ValueError(f"Policy was conditioned on {used!r}, not the requested {self.pending_prompt!r}")
                timings = result.get("timings", {})
                self.decisions.append({"seed": self._seed(len(self.decisions)), "executed": k,
                                       "prompt": self.pending_prompt, "edge": self.pending_edge,
                                       "prompt_choice": self.pending_choice, "model_prompt": used,
                                       "chunk_length": len(chunk),
                                       "planning_s": timings.get("planning_s"),
                                       "worker_planning_s": timings.get("worker_planning_s")})
            elif self.plan_timeout_s is not None and self.clock() - self.submitted_at > self.plan_timeout_s:
                what = "reset" if self.resetting else "plan"
                raise TimeoutError(f"Policy {what} exceeded rollout.plan_timeout_s={self.plan_timeout_s:g} s")
        if not self.queue and self.future is None:
            prompt = self.current_prompt(self.edge)
            self.pending_prompt, self.pending_edge = prompt, self.edge
            self.pending_choice = self.prompt_choice or "auto"
            self.future = self.worker.submit_plan(self.observation(obs, previous, prompt), self._seed(len(self.decisions)))
            self.submitted_at = self.clock()
        if self.queue:
            self.status = "executing"
            action, self.action_prompt = self.queue.popleft()
            return action.copy()
        self.status = "planning"
        return None

    def frame_prompt(self, edge=None):
        """The instruction to label a recorded frame with.

        Policy frames carry the prompt of the plan that produced the executed (or
        held) action. Human and paused frames carry the prompt in effect, so
        corrections are paired with the instruction the policy was following.
        """
        if self.status in {"executing", "planning"}:
            return self.action_prompt or self.pending_prompt
        try:
            return self.current_prompt(None if edge is None else tuple(edge))
        except ValueError:
            return self.prompt

    def cancel(self, status="human"):
        """Drop queued actions immediately; an in-flight plan finishes and is ignored."""
        self.queue.clear()
        if self.future is not None:
            self.stale = self.future
            self.future = None
        self.resetting = False
        self.status = status

    def _seed(self, decision):
        return int(np.random.SeedSequence([self.seed_base, int(decision)]).generate_state(1)[0])

    def summary(self):
        planning = [d["planning_s"] for d in self.decisions if d["planning_s"] is not None]
        return {"checkpoint": self.checkpoint, "config_name": self.config_name, "prompt": self.prompt,
                "edge_prompts": dict(self.prompts) or None, "extra_prompts": dict(self.extra_prompts) or None,
                "execute_k": self.execute_k, "seed_base": self.seed_base, "decisions": len(self.decisions),
                "planning_s_mean": float(np.mean(planning)) if planning else None,
                "planning_s_max": float(np.max(planning)) if planning else None,
                "decision_log": list(self.decisions),
                "worker": {key: self.metadata.get(key) for key in (
                    "proposer_kind", "trained_action_horizon", "candidate_horizon", "observation_horizon",
                    "dependency_versions", "inference_hardware", "python", "policy_cameras")}}

    def snapshot(self):
        last = self.decisions[-1]["planning_s"] if self.decisions else None
        model_prompt = self.decisions[-1]["model_prompt"] if self.decisions else None
        current = None
        if self.edge is not None or self.prompt_choice is not None:
            try:
                current = self.current_prompt(self.edge)
            except ValueError:
                current = None
        return {"available": self.status != "error", "checkpoint": Path(self.checkpoint).name,
                "prompt": current, "model_prompt": model_prompt, "edge_prompts": bool(self.prompts),
                "prompt_choice": self.prompt_choice or "auto", "prompt_options": self.prompt_options(),
                "config_name": self.config_name, "status": self.status, "decisions": len(self.decisions),
                "last_planning_s": last, "execute_k": self.execute_k, "error": self.error}

    def close(self):
        self.worker.close()
