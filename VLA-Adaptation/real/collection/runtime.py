"""Single robot-loop owner and the browser command boundary."""
from __future__ import annotations

from contextlib import ExitStack
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import threading
import time

import numpy as np

from real.collection.guide import Conflict, load_task
from real.collection.recorder import check_writer_dependencies
from real.collection.session import Session
from real.collection.teleop import GripperToggle, SimulatedEnv, taken
from real.joint_limits import reset_target
from real.rotation_utils import homogeneous_to_pose_10d, pose_10d_to_homogeneous
from real.reset_sampling import sample_reset_qpos


class Collection:
    def __init__(self, cfg, task, env, mouse=None, recorder=None, camera_arm=None, policy=None):
        self.cfg, self.task, self.env, self.mouse, self.recorder = cfg, task, env, mouse, recorder
        self.camera_arm = camera_arm
        # Optional real.collection.policy.PolicyDriver: P rolls out, Space takes over.
        self.policy = policy
        self.simulate = bool(cfg.collect.simulate)
        self.dry_run = bool(cfg.collect.dry_run)
        self.auto_reset_after_save = bool(cfg.collect.get("auto_reset_after_save", True))
        self.translation_scale = float(cfg.collect.get("spacemouse_translation_scale", 1.0))
        self.session = Session(task, recorder.counts if recorder else None)
        self.lock = threading.RLock()
        self.pending = None
        self.known_events = set()
        self.owner = None
        self.last_heartbeat = 0.
        self.preview_images = {}
        self.staged = None
        self.tick = 0
        self.motion_ticks = 0
        self.running = True
        self.fatal = False
        self.obs = env._get_obs()
        self.previous_obs = None
        self.desired = env.robot.get_ee_pose()
        self.gripper = GripperToggle(float(cfg.collect.gripper_open_width), float(cfg.collect.gripper_closed_width), env.robot.get_gripper_width())
        self.previous_measured = env.robot.get_state_10d()
        self.last_preview = 0.
        self.reset_rng = np.random.default_rng()
        self._save_executor = None
        self._save_future = None
        self._pending_save = None
        self._preview_executor = None
        self._preview_future = None

    def heartbeat(self, client_id):
        with self.lock:
            if client_id and client_id == self.owner:
                self.last_heartbeat = time.monotonic()

    def snapshot(self):
        with self.lock:
            task = deepcopy(self.task)
            for state in task["states"]:
                state.pop("image_path", None)
            return {"mode": "collection", "collector_connected": True, "simulation": self.simulate,
                    "auto_reset_after_save": self.auto_reset_after_save,
                    "save_reset_state_id": self._save_reset_state(),
                    "save_status": deepcopy(self._pending_save['status']) if self._pending_save else None,
                    "dry_run": self.dry_run, "task": task, "session": self.session.snapshot(),
                    "events": deepcopy(self.session.events[-12:][::-1]), "server_time": time.time(),
                    "command_pending": self.pending is not None, "owner": self.owner,
                    "error": self.session.error, "fatal": self.fatal, "cameras": list(self.preview_images),
                    "rollout": self.policy.snapshot() if self.policy else None}

    def export(self):
        return {"task": self.snapshot()["task"], "session": self.session.metadata(),
                "events": deepcopy(self.session.events), "simulation": self.simulate}

    def command(self, body):
        with self.lock:
            identity, client = body.get("event_id"), body.get("client_id")
            if not isinstance(identity, str) or not 1 <= len(identity) <= 100 or not isinstance(client, str) or not 1 <= len(client) <= 100:
                raise ValueError("event_id and client_id are required")
            if self.owner and client != self.owner:
                abandoned = time.monotonic() - self.last_heartbeat > float(self.cfg.collect.ui.heartbeat_timeout)
                if not abandoned or self.session.mode not in {"setup", "review", "interrupted"}:
                    raise Conflict("Another browser tab owns this collection session")
            if identity in self.known_events:
                return self.snapshot()
            if self.pending or body.get("revision") != self.session.revision:
                raise Conflict("An action is pending or the session changed. Review the current status.")
            if self._pending_save is not None:
                raise Conflict("The previous episode is still saving. Arrange the scene while it finishes.")
            if self.fatal:
                raise ValueError("A device or commit failed. Restart after inspecting the error and draft.")
            if body.get("action") not in {"reset", "teleop", "start", "rollout", "pause", "goal", "undo", "accept",
                                          "discard", "simulate_motion", "prompt", "route"}:
                raise ValueError("Unknown collection action")
            self.owner = client
            self.last_heartbeat = time.monotonic()
            self.pending = {**body, "received_at": time.time(), "received_monotonic_s": time.monotonic()}
            self.known_events.add(identity)
            return self.snapshot()

    def _hold(self):
        """Stop accumulating targets without depending on cameras or disk."""
        self.desired = self.env.robot.hold()

    def _resync(self):
        self._hold()
        self.obs = self.env._get_obs()
        # Keep the commanded gripper target, including force against a grasped object.
        self.gripper.armed = False
        self.previous_measured = self.env.robot.get_state_10d()

    def _stop_policy(self, status="ready"):
        """Stop policy control and attach its summary to the attempt metadata."""
        if self.policy is None:
            return
        if self.session.collection_mode == "rollout":
            self.session.rollout = self.policy.summary()
        if self.policy.status != "error":
            self.policy.cancel(status)

    def _policy_failed(self, exc):
        message = f"Policy stopped: {exc}"
        worker = getattr(self.policy.worker, "process", None)
        dead = worker is not None and worker.poll() is not None
        self._stop_policy("error" if dead else "ready")
        self.policy.error = message
        self._resync()
        self.session.interrupt(message + (" The inference worker exited; restart the collector to roll out again."
                                          if dead else " Discard and retry, or pause it next time."))
        self._checkpoint()

    def _event(self, body):
        return {"id": body["event_id"], "action": body["action"], "at": body["received_at"],
                "at_utc": datetime.fromtimestamp(body["received_at"], timezone.utc).isoformat(),
                "received_monotonic_s": body["received_monotonic_s"], "applied_monotonic_s": time.monotonic(),
                "control_tick_index": self.tick, "attempt_id": self.session.attempt_id,
                "route_index": self.session.route_index}

    def _checkpoint(self):
        if self.staged:
            self.staged.checkpoint(self.session.metadata())

    def _save_reset_state(self):
        """Preview exactly which start E will select, including quota rollover."""
        if self._pending_save is not None:
            return self._pending_save['status']['next_state_id']
        if not self.auto_reset_after_save or self.session.mode not in {"review", "saving"}:
            return None
        counts = self.session.counts.copy()
        if not self.dry_run:
            counts[self.session.route_index] += 1
        index = self.session.next_route(counts)
        return None if index is None else self.task['routes'][index]['states'][0]

    def _reset_to_start(self, trigger="manual"):
        session = self.session
        if session.mode != "setup":
            raise ValueError("Reset is available only during setup")
        session.reset_ready = False
        session.reset_info = None
        reset_target(self.task['states'][session.route[0]]['qpos'])
        session.mode = "resetting"
        session.teleop_enabled = False
        session.revision += 1
        session.reset_info = self._move_to_state(session.route[0], trigger)
        session.reset_ready = True
        session.mode = "setup"

    def _move_to_state(self, state_id, trigger):
        """Robot commands stay on the control loop, including during background saves."""
        nominal = self.task["states"][state_id]["qpos"]
        reset_info = sample_reset_qpos(nominal, self.task.get('qpos_noise_deg', 2.0), self.reset_rng)
        camera_qpos = self.task.get('agilex_qpos')
        if camera_qpos is not None:
            measured = None
            if not self.simulate:
                if self.camera_arm is None:
                    raise RuntimeError('Camera-arm controller is unavailable for the configured agilex_qpos')
                self._hold()
                measured = self.camera_arm.move(np.asarray(camera_qpos))
            reset_info['camera_arm'] = {
                'target_qpos': list(camera_qpos), 'can': self.task.get('agilex_can', 'can_piper'),
                'simulation': self.simulate, 'arrival_verified': measured is not None,
                'measured_qpos': None if measured is None else np.asarray(measured).tolist(),
                'settle_seconds': 1.0}
        self.env.reset(options={"qpos": reset_info['target'], "open_gripper": True})
        self.gripper.target = self.gripper.open_width
        self._resync()
        return {**reset_info, "state_id": state_id, "trigger": trigger,
                "measured": self.env.robot.get_qpos().tolist()}

    def _begin_save(self, event):
        session = self.session
        if session.mode != "review":
            raise ValueError("Reach every goal before saving")
        next_state = self._save_reset_state()
        session.mode = "saving"
        session.teleop_enabled = False
        session.revision += 1
        session.events.append(event)
        metadata = deepcopy(session.metadata())
        episode = self.staged
        camera_shapes = {name: image.shape for name, image in self.obs['images'].items()}
        if episode is not None:
            # Durable source + boundaries before any reset; no PNG/Parquet/MP4 yet.
            episode.checkpoint(metadata)
            episode.flush()
            episode.close()
        self.staged = None
        self._pending_save = {'episode': episode, 'reset': None, 'reset_error': None,
            'status': {'phase': 'resetting' if next_state is not None else 'encoding',
                       'next_state_id': next_state, 'reset_complete': False,
                       'episode_uuid': metadata['attempt_id']}}
        if next_state is not None:
            session.mode = 'resetting'
            session.revision += 1
            try:
                self._pending_save['reset'] = self._move_to_state(next_state, 'before_save')
                self._pending_save['status']['reset_complete'] = True
            except Exception as exc:
                # A good demonstration still gets saved if repositioning fails.
                self._pending_save['reset_error'] = str(exc)
        session.mode = 'saving'
        session.revision += 1
        self._pending_save['status']['phase'] = 'encoding'
        if episode is not None:
            if self._save_executor is None:
                self._save_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='dataset-save')
            self._save_future = self._save_executor.submit(
                self.recorder.commit, episode, metadata, camera_shapes, str(self.cfg.collect.task_prompt))
        else:
            self.finish_save()  # Dry run: reset without a dataset commit.

    def finish_save(self, wait=False):
        """Publish completion on the collector thread; drain on shutdown/tests."""
        pending = self._pending_save
        if pending is None:
            return
        if self._save_future is not None and not wait and not self._save_future.done():
            return
        try:
            if pending['episode'] is not None and self._save_future is None:
                raise RuntimeError('Saving was interrupted before the encoding worker started')
            if self._save_future is not None:
                self._save_future.result()
        except Exception as exc:
            self.fatal = True
            self.session.reset_ready = False
            self.session.interrupt(f"Dataset save failed: {exc}. Staged episode preserved at "
                                   f"{pending['episode'].path}. The reset may already have completed; "
                                   "inspect the dataset before restarting.")
        else:
            self.session.accepted(count=not self.dry_run)
            if pending['reset_error'] is not None:
                self.fatal = True
                prefix = 'Practice finished' if self.dry_run else 'Episode saved successfully'
                self.session.interrupt(f"{prefix}, but automatic reset to state "
                    f"{pending['status']['next_state_id']} failed: {pending['reset_error']}. "
                    "Restart the collector before the next attempt.")
            elif pending['reset'] is not None and self.session.mode == 'setup':
                self.session.reset_info = pending['reset']
                self.session.reset_ready = True
        finally:
            self._save_future = None
            self._pending_save = None
            self.session.revision += 1
        if self.fatal:
            try:
                self._hold()
            except Exception:
                pass

    def close(self):
        self.finish_save(wait=True)
        if self._save_executor is not None:
            self._save_executor.shutdown(wait=True)
            self._save_executor = None
        if self._preview_executor is not None:
            self._preview_executor.shutdown(wait=True)
            self._preview_executor = None

    def _action(self, body):
        session, action = self.session, body["action"]
        event = self._event(body)
        if action == "reset":
            self._reset_to_start()
        elif action == "teleop" and session.mode == "recording":
            # The SpaceMouse takes over from the policy or a pause, from the held pose;
            # queued policy actions are dropped at once.
            session.resume("human")
            self.policy.cancel("human")
            self._resync()
            event.update(controller="human", handoff_frame=session.frames)
        elif action == "teleop":
            if session.mode != "setup":
                raise ValueError("Setup teleoperation is available only during setup")
            self._resync()
            session.teleop_enabled = not session.teleop_enabled
        elif action == "start":
            session.begin(rollout=self.policy is not None)
            if self.policy is not None:
                # The policy may take over later in this attempt; its log starts here.
                self.policy.new_attempt(int(self.reset_rng.integers(2**31)))
            self._resync()
            if self.recorder:
                self.staged = self.recorder.start(session.attempt_id)
            event["attempt_id"] = session.attempt_id
        elif action == "rollout" and session.mode == "recording":
            # Hand the arm to the policy, from the SpaceMouse or a pause, under the prompt now selected.
            if self.policy is None:
                raise ValueError("No policy is loaded; launch the collector with a rollout preset")
            if session.controller == "policy":
                raise ValueError("The policy already drives")
            remaining = None
            if self.policy.prompt_choice is None:
                remaining = session.route[min(len(session.goals), len(session.route) - 2):]
            self.policy.resume(route=remaining)
            session.resume("policy")
            self._resync()
            event.update(controller="policy", handoff_frame=session.frames,
                         prompt_choice=self.policy.prompt_choice or "auto")
        elif action == "rollout":
            if self.policy is None:
                raise ValueError("No policy is loaded; launch the collector with a rollout preset")
            if session.mode != "setup" or not session.reset_ready:
                raise ValueError("Reset to the starting state's qpos before running the policy")
            self.policy.begin(int(self.reset_rng.integers(2**31)), route=session.route)
            session.begin(controller="policy")
            self._resync()
            if self.recorder:
                self.staged = self.recorder.start(session.attempt_id)
            event.update(attempt_id=session.attempt_id, checkpoint=self.policy.checkpoint)
        elif action == "pause":
            session.pause()
            # Drop queued policy actions at once; the arm holds where it is, keeping the gripper command.
            self.policy.cancel("paused")
            self._resync()
            event.update(handoff_frame=session.frames, handoff_time_s=session.handoffs[-1]["time_s"])
        elif action == "route":
            # Dry runs may revisit routes whose quota is complete.
            session.select_route(body.get("route_index"), allow_complete=self.dry_run)
            event.update(route_index=session.route_index, start_state_id=session.route[0])
        elif action == "prompt":
            if self.policy is None:
                raise ValueError("No policy is loaded; launch the collector with a rollout preset")
            remaining = None
            if session.policy_driving:
                remaining = session.route[min(len(session.goals), len(session.route) - 2):]
            self.policy.select_prompt(body.get("prompt_choice"), route=remaining)
            event.update(prompt_choice=self.policy.prompt_choice or "auto",
                         prompt=self.policy.prompt_options().get(self.policy.prompt_choice))
        elif action == "goal":
            event.update(session.goal(event))
            if self.staged:
                arrival = self.env._get_obs()
                np.savez(self.staged.path / f"arrival-{len(session.goals):02d}.npz", state=arrival["state"],
                         **{f"images.{k}": v for k, v in arrival["images"].items()})
            if session.mode == "review":
                self._stop_policy()
                self._resync()
        elif action == "undo":
            event["undone_event_id"] = session.undo()["id"]
            self._resync()
        elif action == "accept":
            self._stop_policy()
            self._begin_save(event)
            return
        elif action == "discard":
            if session.mode not in {"recording", "review", "interrupted"}:
                raise ValueError("No attempt to discard")
            session.events.append(event)
            self._stop_policy()
            self._checkpoint()
            if self.staged:
                self.staged.close()
                self.staged = None
            session.discard()
            self._reset_to_start(trigger="after_discard")
            return
        elif action == "simulate_motion":
            if not self.simulate or session.mode != "recording":
                raise ValueError("Synthetic motion is available only in a simulated recording")
            self.motion_ticks = 10
        session.events.append(event)
        self._checkpoint()

    def process_command(self):
        with self.lock:
            body = self.pending
        if body is None:
            return
        try:
            self.session.error = None
            self._action(body)
        except ValueError as exc:
            if self.fatal or self.session.mode in {"saving", "resetting"}:
                self.fatal = True
                self.session.interrupt(str(exc))
                self._hold()
            else:
                self.session.error = str(exc)
        except Exception as exc:
            self.session.interrupt(str(exc))
            self.fatal = True
            self._hold()
            self._checkpoint()
        finally:
            with self.lock:
                self.pending = None
                self.session.revision += 1

    def step(self):
        self.finish_save()
        self.process_command()
        if self._pending_save is not None or self.fatal:
            # The reset established the hold. Never send robot commands from
            # the writer thread, or keep stepping a robot whose reset failed.
            time.sleep(.02)
            return
        session = self.session
        driving = session.teleop_enabled or session.policy_driving
        # Dry runs keep driving without the page's heartbeat (e.g. paused in a debugger).
        if (driving and self.owner and not self.dry_run
                and time.monotonic() - self.last_heartbeat > float(self.cfg.collect.ui.heartbeat_timeout)):
            self._stop_policy()
            self._resync()
            session.interrupt("Operator connection lost; draft preserved. Retry after reconnecting.")
            self._checkpoint()
        if session.policy_driving and not self.fatal:
            try:
                segment = min(len(session.goals), len(session.route) - 2)
                planned = self.policy.next_action(self.obs, self.previous_obs,
                                                  edge=session.route[segment:segment + 2])
            except Exception as exc:
                self._policy_failed(exc)
                planned = None
            if planned is not None:
                # Policy targets are absolute and executed unchanged; keeping them as the
                # desired pose makes a pause hold the last policy command.
                self.desired, width = pose_10d_to_homogeneous(planned)
                self.gripper.target = width
        elif session.teleop_enabled and not self.fatal:
            if self.simulate:
                delta, buttons = np.eye(4), ()
                if self.motion_ticks:
                    delta[0, 3] = .002
                    self.motion_ticks -= 1
            else:
                delta, buttons = self.mouse.snapshot()
            self.desired[:3, 3] += delta[:3, 3] * self.translation_scale
            self.desired[:3, :3] = delta[:3, :3] @ self.desired[:3, :3]
            self.gripper.update(buttons)
        action = homogeneous_to_pose_10d(self.desired, self.gripper.target)
        current = self.env.robot.get_state_10d()
        # Preserve the original observation/action pairing and comparison order.
        valid = taken(self.previous_measured, current)
        frame = None
        if session.mode == "recording" and valid:
            frame = {"observation.state": np.asarray(self.obs["state"], dtype=np.float32), "action": action,
                     "timing.capture_elapsed_s": np.array([self.obs.get("capture_monotonic_s", time.monotonic()) - session.started_monotonic], dtype=np.float64),
                     "timing.control_tick_index": np.array([self.tick], dtype=np.int64)}
            frame.update({f"observation.images.{name}": image for name, image in self.obs["images"].items()})
            if self.policy is not None:
                frame["intervention"] = np.array([0 if session.controller == "policy" else 1], dtype=np.int64)
                # Becomes this frame's LeRobot task: the prompt that generated the action.
                segment = min(len(session.goals), len(session.route) - 2)
                frame["task"] = np.array(self.policy.frame_prompt(session.route[segment:segment + 2]))
        policy_frame = session.controller == "policy"
        self.previous_measured = current.copy()
        self.previous_obs = self.obs
        self.obs, _, _, _, info = self.env.step(action)
        # Feed back the accepted reference: a shortened move must not build
        # a hidden backlog of targets.
        applied = np.asarray(info["applied_action"], dtype=np.float32)
        if info.get("velocity_scale", 1.) < 1.:  # Shortened to the joint velocity limits.
            self.desired, _ = pose_10d_to_homogeneous(applied)
        if frame is not None:
            frame["action"] = applied.copy()
            if self.staged:
                self.staged.add(frame)
            session.frames += 1
            session.policy_frames += int(policy_frame)
        self.tick += 1
        preview_idle = self._preview_future is None or self._preview_future.done()
        if preview_idle and time.monotonic() - self.last_preview > .1:
            # Encode in a thread so previews never delay the next control tick.
            if self._preview_executor is None:
                self._preview_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='preview')
            self._preview_future = self._preview_executor.submit(self._encode_previews, self.obs["images"])
            self.last_preview = time.monotonic()

    def _encode_previews(self, images):
        import cv2
        previews = {}
        for name, image in images.items():
            ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 75])
            if ok:
                previews[name] = encoded.tobytes()
        with self.lock:
            self.preview_images = previews

    def run(self):
        try:
            while self.running:
                try:
                    self.step()
                except Exception as exc:
                    self.session.interrupt(str(exc))
                    self.fatal = True
                    try:
                        self._hold()
                    except Exception:
                        pass
                    try:
                        self._checkpoint()
                    except Exception as storage_error:
                        self.session.error += f"; draft checkpoint failed: {storage_error}"
                    while self.running:
                        time.sleep(.1)
        finally:
            self.close()
            try:
                self._hold()
            except Exception:
                pass
            if self.staged:
                self.session.interrupt("Collector stopped; draft preserved")
                self._checkpoint()
                self.staged.close()


def run(cfg):
    task = validated_task(cfg)
    try:
        main(cfg, task)
    except KeyboardInterrupt:
        pass


def validated_task(cfg):
    """Check the configuration and task before opening any device."""
    from hydra.core.hydra_config import HydraConfig
    from omegaconf import OmegaConf
    if "arms" in cfg:
        raise ValueError("SFT collection supports one robot; use collect.yaml")
    if float(cfg.collect.fps) <= 0 or int(cfg.collect.fps) != float(cfg.collect.fps) or float(cfg.collect.fps) != float(cfg.robot.freq):
        raise ValueError("collect.fps must equal robot.freq for the nominal retained-frame timeline")
    if float(cfg.collect.spacemouse_freq) < float(cfg.collect.fps):
        raise ValueError("collect.spacemouse_freq must be at least collect.fps")
    translation_scale = float(cfg.collect.get("spacemouse_translation_scale", 1.0))
    if not np.isfinite(translation_scale) or not 0 < translation_scale <= 1:
        raise ValueError("collect.spacemouse_translation_scale must be greater than 0 and at most 1")
    task_name = HydraConfig.get().runtime.choices.get("task", "tabletop") if HydraConfig.initialized() else "tabletop"
    task_path = Path(__file__).resolve().parents[1] / "config/task" / f"{task_name}.yaml"
    task = load_task(task_path, OmegaConf.to_container(cfg.task, resolve=True))
    for state in task["states"]:
        if not state["image_url"]:
            raise ValueError(f"State {state['id']} needs a reference image")
        # A hardware dry run can exercise setup teleop before poses are captured.
        # Explicit reset commands still validate their target in _action().
        if not cfg.collect.simulate and not cfg.collect.dry_run:
            try:
                reset_target(state["qpos"])
            except ValueError as exc:
                raise ValueError(f"Capture state {state['id']}'s qpos first: {exc}") from exc
    if not cfg.collect.dry_run:
        check_writer_dependencies()
        str(cfg.collect.task_prompt)  # Validate the required field before opening devices.
    return task


def main(cfg, task):
    from omegaconf import OmegaConf
    from real.collection.server import make_server
    simulate = bool(cfg.collect.simulate)
    with ExitStack() as stack:
        recorder = None
        if not cfg.collect.dry_run:
            # LeRobot writes in its own process, started before the robot and the model.
            from real.collection.save_worker import ProcessRecorder
            recorder = ProcessRecorder(Path(cfg.collect.dataset_root).expanduser() / str(cfg.collect.repo_id),
                str(cfg.collect.repo_id), task, int(cfg.collect.fps), bool(cfg.collect.use_videos),
                int(cfg.collect.image_writer_threads), simulate,
                encoding=OmegaConf.to_container(cfg.collect.video_encoding, resolve=True),
                intervention=cfg.get("rollout") is not None, realtime_cpu=cfg.robot.get("realtime_cpu"),
                **OmegaConf.to_container(cfg.collect.save, resolve=True))
            stack.callback(recorder.close)
            print(f"Dataset writer ready in process {recorder.worker['pid']}", flush=True)
        if not simulate:
            # Reserve the arm before the model loads, so a second collector fails fast.
            from real.robot import acquire_control
            stack.callback(acquire_control(cfg.robot.ip).close)
        policy = None
        if cfg.get("rollout") is not None:
            # Load and compile the policy before taking torque control, so model
            # loading never competes with a freshly started controller.
            from real.collection.policy import PolicyDriver, start_worker
            worker = start_worker(cfg)
            stack.callback(worker.close)
            policy = PolicyDriver.from_config(cfg, worker)
        if simulate:
            env = SimulatedEnv(float(cfg.robot.freq), cfg.task.perception.keys())
        else:
            from real.env import FR3RealEnv
            env = FR3RealEnv(cfg, task="dry run" if cfg.collect.dry_run else str(cfg.collect.task_prompt))
        stack.callback(env.close)
        env.start()
        mouse = None
        if not simulate:
            from real.spacemouse import SpaceMouseController
            mouse = SpaceMouseController(freq=float(cfg.collect.spacemouse_freq), path=cfg.collect.spacemouse_path)
            stack.callback(mouse.close)
        camera_arm = None
        if not simulate and task.get('agilex_qpos') is not None:
            from real.agilex import AgileXPositionController
            camera_arm = AgileXPositionController(can=task['agilex_can'])
            stack.callback(camera_arm.close)
        collection = Collection(cfg, task, env, mouse, recorder, camera_arm=camera_arm, policy=policy)
        stack.callback(collection.close)
        server = make_server(collection, str(cfg.collect.ui.host), int(cfg.collect.ui.port))
        stack.callback(server.server_close)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        stack.callback(server.shutdown)
        print(f"Single-FR3 collector: http://localhost:{cfg.collect.ui.port}", flush=True)
        if cfg.collect.dry_run:
            print("Dry run: no plan, worker or page-heartbeat timeouts. Closing the page does not stop "
                  "the policy; pause it with Space or use the robot's stop.", flush=True)
        print("H: reset to route start; T: setup teleop; S: record; G: goal; E: save. "
              + ("P: run or resume policy; Space: pause; T: teleop after a pause. " if policy else "")
              + ("E resets to the next start before background encoding. Clear the arm's path before E."
                 if collection.auto_reset_after_save else "Use H to reset after saving."), flush=True)
        collection.run()
