"""Single-arm route boundaries, independent of input devices and storage."""
from copy import deepcopy
import time
import uuid


class Session:
    def __init__(self, task, counts=None):
        self.task = task
        self.counts = list(counts or [0] * len(task["routes"]))
        self.revision = 0
        self.route_index = 0
        # Operator's route choice (and so its start state); None follows quota order.
        self.selected_route = None
        self.events = []
        self.error = None
        self._setup()

    def next_route(self, counts=None):
        """The selected route while its quota is open, otherwise the first open route."""
        counts = self.counts if counts is None else counts
        routes = self.task["routes"]
        if self.selected_route is not None and counts[self.selected_route] < routes[self.selected_route]["quota"]:
            return self.selected_route
        return next((i for i, route in enumerate(routes) if counts[i] < route["quota"]), None)

    def select_route(self, index, allow_complete=False):
        """Choose the route (and so the start state) for the next attempts, during setup."""
        if self.mode != "setup":
            raise ValueError("Choose a start state during setup")
        if type(index) is not int or not 0 <= index < len(self.task["routes"]):
            raise ValueError(f"Unknown route {index!r}")
        if not allow_complete and self.counts[index] >= self.task["routes"][index]["quota"]:
            raise ValueError(f"Sequence {index + 1}'s quota is complete")
        if index != self.route_index:
            self.reset_ready = False  # The arm is at the previous route's start.
            self.reset_info = None
        self.selected_route = self.route_index = index

    def _setup(self):
        next_route = self.next_route()
        self.mode = "complete" if next_route is None else "setup"
        if next_route is not None:
            self.route_index = next_route
        self.attempt_id = None
        self.started_at = None
        self.started_monotonic = None
        self.goals = []
        self.frames = 0
        self.reset_ready = False
        self.teleop_enabled = False
        self.reset_info = None
        # "human" (SpaceMouse), "policy", or "paused" (the arm holds). A rollout can pause
        # and hand control to the human or back to the policy any number of times.
        self.controller = "human"
        self.collection_mode = "teleop"
        # The first pause of a rollout: where the policy first stopped driving.
        self.takeover_frame = None
        self.takeover_time_s = None
        self.handoffs = []
        self.policy_frames = 0
        self.rollout = None

    @property
    def route(self):
        return self.task["routes"][self.route_index]["states"]

    def begin(self, controller="human", rollout=False):
        """Start recording. rollout (a policy is loaded) lets the policy, a pause and the
        SpaceMouse take turns in this attempt, whoever starts it."""
        if self.mode != "setup" or not self.reset_ready:
            raise ValueError("Reset to the starting state's qpos before recording")
        if controller not in {"human", "policy"}:
            raise ValueError(f"Unknown controller {controller!r}")
        self.mode = "recording"
        self.attempt_id = str(uuid.uuid4())
        self.started_at = time.time()
        self.started_monotonic = time.monotonic()
        self.frames = 0
        self.goals = []
        self.controller = controller
        self.collection_mode = "rollout" if rollout or controller == "policy" else "teleop"
        self.takeover_frame = None
        self.takeover_time_s = None
        self.handoffs = []
        self.policy_frames = 0
        self.teleop_enabled = controller == "human"

    def _hand_to(self, controller):
        elapsed = time.monotonic() - self.started_monotonic
        self.handoffs.append({"controller": controller, "frame": self.frames, "time_s": elapsed})
        # The first takeover from the policy: a pause, or the SpaceMouse taking over directly.
        if self.takeover_frame is None and (controller == "paused" or self.controller == "policy"):
            self.takeover_frame, self.takeover_time_s = self.frames, elapsed
        self.controller = controller
        self.teleop_enabled = controller == "human"

    def pause(self):
        """Stop the policy or the human mid-rollout; the arm holds until resume()."""
        if self.mode != "recording" or self.collection_mode != "rollout" or self.controller == "paused":
            raise ValueError("Pause is available while the policy or the operator drives a rollout recording")
        self._hand_to("paused")

    def resume(self, controller):
        """Hand a rollout recording to the SpaceMouse ("human") or the policy, from either
        the other one or a pause."""
        if self.mode != "recording" or self.collection_mode != "rollout":
            raise ValueError("Handing over is available during a rollout recording")
        if controller not in {"human", "policy"}:
            raise ValueError(f"Unknown controller {controller!r}")
        if controller == self.controller:
            raise ValueError("The policy already drives" if controller == "policy" else "You already drive")
        self._hand_to(controller)

    def goal(self, event):
        if self.mode != "recording":
            raise ValueError("Goal markers require an active recording")
        previous_boundary = self.goals[-1]["boundary_index"] if self.goals else 0
        if self.frames <= previous_boundary:
            raise ValueError("This edge has no taken transitions; move before marking its goal")
        segment = len(self.goals)
        marker = {**event, "boundary_index": self.frames, "edge": self.route[segment:segment + 2],
                  "reached_state_id": self.route[segment + 1], "segment_index": segment}
        self.goals.append(marker)
        if len(self.goals) == len(self.route) - 1:
            self.mode = "review"
            self.teleop_enabled = False
        return marker

    @property
    def policy_driving(self):
        return self.mode == "recording" and self.controller == "policy"

    def undo(self):
        if self.mode not in {"recording", "review"} or not self.goals:
            raise ValueError("No goal to undo")
        undone = self.goals.pop()
        self.mode = "recording"
        self.teleop_enabled = self.controller == "human"
        return undone

    def accepted(self, count=True):
        if self.mode not in {"review", "saving"}:
            raise ValueError("Complete the route before saving")
        if count:
            self.counts[self.route_index] += 1
        self._setup()

    def discard(self):
        if self.mode not in {"recording", "review", "interrupted"}:
            raise ValueError("No attempt to discard")
        self._setup()

    def interrupt(self, message):
        self.mode = "interrupted"
        self.teleop_enabled = False
        self.error = str(message)
        self.revision += 1

    def snapshot(self):
        return {"revision": self.revision, "mode": self.mode, "route_index": self.route_index,
                "counts": self.counts.copy(), "attempt_id": self.attempt_id, "started_at": self.started_at,
                "segment_index": min(len(self.goals), len(self.route) - 2), "goals": deepcopy(self.goals),
                "frames": self.frames, "reset_ready": self.reset_ready, "teleop_enabled": self.teleop_enabled,
                "controller": self.controller, "collection_mode": self.collection_mode,
                "takeover_frame": self.takeover_frame, "policy_frames": self.policy_frames,
                "handoffs": deepcopy(self.handoffs)}

    def metadata(self):
        result = self.snapshot()
        result.update(route=self.route, task_hash=self.task["fingerprint"], accepted=False,
                      route_completed=self.mode in {"review", "saving"}, reset=self.reset_info,
                      events=deepcopy([e for e in self.events if e.get("attempt_id") == self.attempt_id]),
                      takeover_time_s=self.takeover_time_s, rollout=deepcopy(self.rollout))
        start = 0
        result["segments"] = []
        for marker in self.goals:
            result["segments"].append({"edge": marker["edge"], "start": start, "end": marker["boundary_index"]})
            start = marker["boundary_index"]
        return result
