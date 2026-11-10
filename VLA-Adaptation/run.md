# Deploy pi0.5 kitchen SFT v5 (h50 canonical, step 22000)

`rollout=pi05_kitchen_sft_v5` loads the checkpoint from the git-ignored `checkpoints/` folder
and executes all 50 actions of each chunk before replanning (`rollout.execute_k=50`).

## 1. Download the weights and link the norm stats

From `VLA-Adaptation/`:

```bash
conda run -n rsm python -c "from huggingface_hub import snapshot_download; snapshot_download('xinyunsunshine/pi05-kitchen-sft-v5', allow_patterns=['h50/canonical/checkpoints/22000/**', 'h50/canonical/*.json', 'h50/canonical/README.md'], local_dir='checkpoints/pi05-kitchen-sft-v5')"
C=checkpoints/pi05-kitchen-sft-v5/h50/canonical/checkpoints/22000
mkdir -p $C/assets/jmarangola && ln -sfn ../local/rsm_cooking_v5_language $C/assets/jmarangola/rsm_cooking_v5
```

## 2. Run on the robot

Run `aiofranka unlock` first. Add `collect.dry_run=true` for an unrecorded first trial.

```bash
conda run --no-capture-output -n adaptive python -m real.collect_sft_data task=tabletop \
  rollout=pi05_kitchen_sft_v5 \
  collect.dataset_root=/home/ankile/lrds collect.repo_id=DAgger_part1 \
  collect.task_prompt="Follow the displayed cooking state sequence"
```

## 3. Evaluate a chosen segment (any start state → any goal state)

The dashboard runs the routes in `task.edge_combinations` in order, starting with the first
route whose quota isn't met. Override that list to hold only the route you want. Set
`START`, `GOAL` and `N` (attempts), then run:

```bash
START=2 GOAL=3 N=20
conda run --no-capture-output -n adaptive python -m real.collect_sft_data task=tabletop \
  rollout=pi05_kitchen_sft_v5 \
  "task.edge_combinations=[[[$START,$GOAL],$N]]" \
  "task.reward_states=[$GOAL]" \
  collect.dataset_root=/home/ankile/lrds collect.repo_id=rsm_cooking_rollouts_canonical_${START}to${GOAL} \
  collect.task_prompt="Follow the displayed cooking state sequence"
```

Arrange the scene to match the start state, press **H** (arm moves to the start state's pose),
then **P** to run the policy, **G** at the goal, **E** to save.

- Multi-step route: `"task.edge_combinations=[[[0,1,3],$N]]"` (press **G** at each goal).
- Several segments in one session: `"task.edge_combinations=[[[1,3],20],[[2,3],20]]"`.
- Use a new `repo_id` for each route list: the dataset is tied to its task config.
- Trained edges only: 0→1, 0→2, 0→5, 1→3, 2→3, 2→6, 3→4.
- Add `collect.simulate=true collect.dry_run=true` to check the command without the robot.

## 4. Switch the prompt during a rollout

Use the **Policy prompt** dropdown under the robot controls:

- **Auto**: follows the route edge; the prompt changes at each **G**.
- **An edge prompt** (e.g. `1 → 3 · Place the pan …`) or **Default**: pins that prompt until
  you change it, across attempts too. Works on edges without a trained prompt.
- Changes apply from the next policy plan; already queued actions (up to `execute_k`) finish first.
  **Space** still pauses at once.

Each rollout frame stores the prompt that generated its action as its LeRobot `task`; human
and paused frames store the prompt in effect. Prompt changes are logged as `prompt` events.

Check it without the robot first (real checkpoint, simulated arm, nothing recorded):

```bash
conda run --no-capture-output -n adaptive python -m real.collect_sft_data task=tabletop \
  rollout=pi05_kitchen_sft_v5 \
  collect.simulate=true collect.dry_run=true
```
