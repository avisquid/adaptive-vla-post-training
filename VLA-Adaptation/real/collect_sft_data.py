"""Single-FR3 SpaceMouse collection with browser subgoals and LeRobot output.

Use conda env adaptive. Unlock the robot (`aiofranka unlock`), then:
    python -m real.collect_sft_data collect.dataset_root=/data \
        collect.repo_id=tabletop_v1 collect.task_prompt="Follow the tabletop route"

Hardware-free browser preview:
    python -m real.collect_sft_data collect.simulate=true collect.dry_run=true

Browser keys: H reset to route start; T setup teleop; S record; G subgoal;
U undo; E save and reset to the next start; R discard and reset. One SpaceMouse
controls the arm and toggles the gripper. Only taken transitions are stored.
"""
import hydra
from omegaconf import DictConfig

from real.collection.runtime import run


@hydra.main(version_base=None, config_path="config", config_name="collect")
def main(cfg: DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
