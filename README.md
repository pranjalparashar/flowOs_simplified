---
title: FlowOs V2
emoji: 📚
colorFrom: red
colorTo: green
sdk: docker
pinned: false
---

# FlowOS V2

FlowOS V2 is an OpenEnv environment for training LLM agents to build and repair realistic data-platform workflows. Instead of solving a toy grid world, the agent has to read workspace files, inspect schemas, write pipeline YAML and SQL, run validators, interpret runtime failures, and submit a working warehouse report.

- Hugging Face Space: [praanjal-flowos-v2.hf.space](https://praanjal-flowos-v2.hf.space)
- OpenEnv manifest: [openenv.yaml](/Users/pranjalparashar/Desktop/hackz/cw-apr1/control-room/developer_control_room/openenv.yaml)
- Training scripts:
  - [collect_traces.py](/Users/pranjalparashar/Desktop/hackz/cw-apr1/control-room/developer_control_room/collect_traces.py)
  - [train_sft.py](/Users/pranjalparashar/Desktop/hackz/cw-apr1/control-room/developer_control_room/train_sft.py)
  - [train.py](/Users/pranjalparashar/Desktop/hackz/cw-apr1/control-room/developer_control_room/train.py)
  - [eval.py](/Users/pranjalparashar/Desktop/hackz/cw-apr1/control-room/developer_control_room/eval.py)
- Plotting script: [plot_metrics.py](/Users/pranjalparashar/Desktop/hackz/cw-apr1/control-room/developer_control_room/plot_metrics.py)
- Blog / writeup: `ADD_HF_BLOG_LINK_HERE`
- Video / slides: `ADD_VIDEO_OR_SLIDES_LINK_HERE`

## Problem

LLM agents still struggle with one of the most common production engineering loops:

1. understand a partially specified data task,
2. create the right pipeline artifacts,
3. run the workflow,
4. interpret runtime failures,
5. repair the broken pieces without violating downstream contracts.

Most existing agent benchmarks either stop at code generation or use abstract environments with low-stakes actions. FlowOS instead targets a concrete capability gap: can an LLM learn to behave like a junior-to-mid-level data platform engineer working in a shared production workspace?

We care about this because modern analytics teams constantly perform exactly this pattern:
- generate ingestion or reporting pipelines from imperfect specs,
- debug warehouse jobs and broken transforms,
- preserve schema contracts while fixing failures,
- coordinate between different responsibilities like “builder” and “fixer”.

## Environment

FlowOS V2 is built on the latest OpenEnv stack and exposed as a remote environment on Hugging Face Spaces.

The current submission focuses on a simulation task family called `simulate_csv_report_workflow`. Each episode places the agent in a small but realistic workspace with:

- a source CSV,
- runtime instructions,
- schema files,
- editable pipeline targets:
  - `pipelines/report_job.yaml`
  - `sql/load_raw.sql`
  - `sql/build_table.sql`
  - `sql/report_view.sql`

The environment then runs those artifacts against a local DuckDB-backed execution runtime and returns:

- runtime success / failure,
- execution logs,
- output schema,
- preview rows,
- validator results.

### What the agent sees

On each step, the agent receives:

- the developer request,
- workspace summary,
- known files,
- editable targets,
- visible schema assets,
- validator names,
- current role (`builder` or `fixer`),
- runtime feedback from prior edits.

### What the agent can do

The agent can:

- `read_file`
- `inspect_schema`
- `edit_file`
- `run_validator`
- `submit_workspace`

### What gets rewarded

The reward and grading logic combines:

- investigation quality,
- artifact completeness,
- runtime success,
- final schema correctness,
- summary quality,
- successful submission.

This is implemented through OpenEnv-compatible state, validator, and grader flows in:

- [tasks.py](/Users/pranjalparashar/Desktop/hackz/cw-apr1/control-room/developer_control_room/tasks.py)
- [graders.py](/Users/pranjalparashar/Desktop/hackz/cw-apr1/control-room/developer_control_room/graders.py)
- [runtime.py](/Users/pranjalparashar/Desktop/hackz/cw-apr1/control-room/developer_control_room/runtime.py)
- [server/developer_control_room_environment.py](/Users/pranjalparashar/Desktop/hackz/cw-apr1/control-room/developer_control_room/server/developer_control_room_environment.py)

## Scenarios

The simulation task currently has 7 deterministic scenarios:

### Pipeline generation

- `SIM-001`: build a customer daily sales report
- `SIM-002`: build a subscription daily report
- `SIM-003`: build an inventory movement report

### Pipeline repair

- `SIM-004`: repair a broken customer-order build SQL
- `SIM-005`: repair a broken subscription report contract
- `SIM-006`: repair a broken inventory report schema
- `SIM-007`: fix a builder-created returns pipeline after runtime failure

This mix was chosen to test both:

- first-pass workflow construction,
- and second-pass repair under runtime feedback.

## Why this environment is interesting

We think this environment is novel because it sits between code generation and full production ops:

- it is not just “write SQL”,
- it is not just “fix a syntax bug”,
- and it is not a toy planning problem.

The agent has to coordinate file creation, schema understanding, runtime-aware debugging, and role-aware behavior inside a shared workspace. That makes it a better fit for training practical engineering agents than many one-shot coding benchmarks.

## Training pipeline

We currently support two training directions:

### 1. Stable path: supervised fine-tuning

This is the main path we are using for reproducible training evidence.

1. Collect teacher traces from the environment:

```bash
python collect_traces.py \
  --env-url https://praanjal-flowos-v2.hf.space \
  --task-scope simulate_csv_report_workflow \
  --dataset-size 28 \
  --max-turns 12 \
  --output-dir outputs/sft_traces_v2
```

2. Train a LoRA SFT policy:

```bash
CUDA_VISIBLE_DEVICES=0 python train_sft.py \
  --dataset-path outputs/sft_traces_v2/traces.jsonl \
  --model-id Qwen/Qwen2.5-1.5B-Instruct \
  --output-dir outputs/flowos-sft-v2 \
  --num-epochs 2 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 4 \
  --max-seq-length 1024 \
  --load-in-4bit \
  --gradient-checkpointing
```

### 2. Experimental path: GRPO / TRL

We also have an RL-style training entrypoint in [train.py](/Users/pranjalparashar/Desktop/hackz/cw-apr1/control-room/developer_control_room/train.py) for future work, but the current submission story is centered on the stable SFT path because it is the most reproducible on commodity GPUs.

## Results

The main question for this environment is simple:

**Does a trained policy behave better than an untrained one?**

To answer that, we compare:

- the base model,
- the fine-tuned checkpoint,
- and optionally the same evaluation with fallback disabled for a stricter policy-only check.

### Evaluation command

```bash
python eval.py \
  --model-id Qwen/Qwen2.5-1.5B-Instruct \
  --checkpoint-path outputs/flowos-sft-v2 \
  --env-url https://praanjal-flowos-v2.hf.space \
  --task-scope simulate_csv_report_workflow \
  --episodes 7 \
  --max-turns 12 \
  --results-dir outputs/eval_compare_v2
```

### Plot generation

```bash
python plot_metrics.py \
  --train-metrics outputs/flowos-sft-v2/training_metrics.csv \
  --eval-metrics outputs/eval_compare_v2/eval_episode_metrics.csv \
  --output-dir outputs/plots_v2
```

### Current evidence

The repository now saves:

- training loss curves,
- per-episode evaluation metrics,
- average reward / score / solved-rate comparisons,
- fallback-step and valid-model-step comparisons,
- generated pipeline artifacts for inspection.

Replace the placeholders below with the final PNGs from your latest run:

- `ADD_TRAINING_LOSS_PLOT_HERE`
- `ADD_EVAL_SCORE_PLOT_HERE`
- `ADD_EVAL_SOLVED_RATE_PLOT_HERE`
- `ADD_BASELINE_VS_TRAINED_TABLE_HERE`

Recommended final README embed format:

```md
![SFT training loss](assets/plots/sft_training_loss.png)
![Average score by policy](assets/plots/eval_avg_score.png)
![Solved rate by policy](assets/plots/eval_solved_rate.png)
```

## How to try the environment

### Run the Space locally

```bash
uv run server
```

### Run a quick environment check

```bash
python -c "from developer_control_room.server.developer_control_room_environment import DeveloperControlRoomEnvironment as E; env=E(); obs=env.reset(task_id='simulate_csv_report_workflow', scenario_index=0); print(obs.scenario_id, obs.active_role, obs.known_files)"
```

### Run evaluation

```bash
python eval.py \
  --model-id Qwen/Qwen2.5-1.5B-Instruct \
  --env-url https://praanjal-flowos-v2.hf.space \
  --task-scope simulate_csv_report_workflow \
  --episodes 1 \
  --max-turns 12 \
  --debug-actions
```

## Submission materials

Please link all final materials here before submission:

- Hugging Face Space: [https://praanjal-flowos-v2.hf.space](https://praanjal-flowos-v2.hf.space)
- Hugging Face blog: `ADD_HF_BLOG_LINK_HERE`
- Video / slides: `ADD_VIDEO_OR_SLIDES_LINK_HERE`
- Colab notebook: `ADD_COLAB_LINK_HERE`

## Future multi-agent architecture

The current environment already has the beginnings of role-aware coordination through the `builder` / `fixer` split. If we extend this into a fuller multi-agent setup with specialized agents like:

- database agent,
- deployment agent,
- pipeline authoring agent,
- incident / fixer agent,

then the architecture should change in a few important ways.

### 1. Role-specific observations and action spaces

Today, all roles see roughly the same workspace shape. In a fuller multi-agent system we would want:

- role-specific observations,
- role-specific allowed actions,
- role-specific reward components.

Example:
- database agent should inspect schemas, SQL, and warehouse outputs,
- deployment agent should inspect configs, schedules, and orchestrator state,
- fixer agent should receive richer runtime and incident logs.

### 2. Shared workspace plus shared memory

We already have a shared workspace. A multi-agent version would also need explicit shared memory for:

- handoff notes,
- failure summaries,
- pending tasks,
- ownership / lock state for files or resources.

That prevents agents from overwriting one another blindly and makes coordination trainable.

### 3. Agent routing / orchestration layer

We would need a coordinator that decides:

- which specialist acts next,
- whether the current role should continue,
- when to escalate from builder to fixer,
- when to stop and submit.

This could be:

- a learned scheduler,
- a rule-based router,
- or another LLM agent acting as a planner.

### 4. Role-conditioned reward design

The reward function should separate:

- local role success,
- global workflow success,
- coordination quality.

For example:
- builder gets rewarded for producing a coherent first-pass pipeline,
- fixer gets rewarded for resolving runtime failures quickly,
- coordinator gets rewarded for minimizing unnecessary turns and handoffs.

### 5. Runtime modularization

As specialized agents grow, the runtime should stop looking like a single flat executor and instead expose modular subsystems:

- warehouse / DuckDB runtime,
- deployment / orchestrator runtime,
- validator runtime,
- artifact registry.

That would let different specialist agents act on different subsystems while still contributing to a shared final score.

### 6. Explicit communication channels

If we want the multi-agent behavior to be a real benchmark and not just sequential role labels, we should add:

- message passing or notes between agents,
- structured handoff actions,
- maybe role-specific submission or escalation actions.

That would make coordination itself part of what gets trained.

## Why this matters

Production AI agents will not succeed only by writing one perfect file. They will need to:

- work in partially broken systems,
- preserve contracts under pressure,
- coordinate across roles,
- and improve through training rather than prompt tricks alone.

FlowOS V2 is our attempt to make that trainable in a compact but realistic OpenEnv environment.
