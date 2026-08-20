# Plan: DreamerV2 → PEG → MoReFree in PyTorch (CREMMA)

Goal: a readable PyTorch reimplementation of MoReFree inside CREMMA, built in layers,
each independently testable, each conforming to `src/algorithms/basepolicy.py`.

**Strategy: adapt, don't reimplement.** We adopt `RajGhugare19/dreamerv2` (vendored at
`dreamerv2/`, 1513 LOC, PyTorch) as the DreamerV2 core, port it to continuous actions +
ContinualBench, then build the PEG and MoReFree layers on top ourselves.

Reference code vendored in this repo:
- `dreamerv2/` — RajGhugare19 PyTorch DreamerV2. **Our base.** MIT, has its own `.git`.
- `peg/dreamerv2/` — PEG (ICLR 2023), TF 2.4. Fork of danijar's DreamerV2 + LEXA. Semantics reference.
- `MoReFree/dreamerv2/` — MoReFree (TMLR 2025), TF 2.4. Fork of PEG. Semantics reference.

---

## 0. Findings that shape the design (verified against the vendored source)

### From PEG / MoReFree (what we must eventually match)

**All observations are state-based. No CNNs.**
Every task config in both `peg/dreamerv2/configs.yaml` and `MoReFree/dreamerv2/configs.yaml`
sets `encoder/decoder: {mlp_keys: 'observation', cnn_keys: '$^', mlp_layers: [400,400,400]}`.
`cnn_keys: '$^'` is a regex matching nothing — the CNN branch is dead in every task. PEG's
README confirms: "State-input version of the LEXA agent, which was designed for image-input
tasks." `image_input: True` in `defaults:` is inherited-then-overridden; pixels are used only
for video logging. ContinualBench is likewise state-based.

**Latents are Gaussian, not categorical.** The `defaults:` block says `discrete: 32` (stock
DreamerV2 categorical), but *every* task override in both repos uses
`rssm: {ensemble: 1, hidden: 200, deter: 200, stoch: 50, discrete: 0, std_act: sigmoid2, min_std: 0.1}`
(`fetch_reach` uses `hidden/deter: 1024`). So the Gaussian path is what we need.

**`ensemble: 1` everywhere.** `EnsembleRSSM`'s ensemble exists only to give Plan2Explore its
disagreement signal; with `ensemble=1` it is a no-op. We add an ensemble later, for PEG.

**MoReFree is a thin layer on PEG.** `diff peg/dreamerv2/gc_agent.py MoReFree/dreamerv2/gc_agent.py`
is ~100 lines: adds `sample_env_init`, `train_env_goal_percent` goal mixing,
`pseudo_goal_sampling`, threads `outputs` out of `train()`. Confirms the layering below.

**Landmine:** `MoReFree/dreamerv2/gc_agent.py:214` has a live `import ipdb; ipdb.set_trace()`
in the reward-head branch. Unreachable under their released configs (`pred_reward` off for GC
training). Do not port that branch assuming it was tested.

### From the base repo (what we get for free vs. must fix)

**Already correct and reusable:**
- `models/rssm.py` + `utils/rssm.py` — RSSM with a working `rssm_type: 'continuous'` branch
  (`RSSMContState(mean, std, stoch, deter)`), which is exactly the Gaussian path PEG/MoReFree use.
- `models/dense.py` `DenseModel` (43 lines) — MLP with `normal`/`binary`/`None` heads; serves as
  encoder, decoder, reward head, discount head, and critic.
- `trainer._model_initialize` already branches on `config.pixel`: **`pixel=False` builds
  `DenseModel` encoder/decoder.** The state-based path exists.
- `trainer._actor_loss:191` already implements **`actor_grad: 'dynamics'`** (backprop through
  the model) alongside `reinforce`.
- KL balancing + free nats (`_kl_loss`), slow-target critic (`update_target`), λ-returns
  (`utils/algorithm.compute_return`), `FreezeParameters`.
- Explicit `for t in range(...)` loops instead of TF `static_scan`; `namedtuple` states.

**Must fix (see phases):**
1. **Actions are discrete-only** — `models/actor.py DiscreteActionModel`, `dist: 'one_hot'`,
   everything else `raise NotImplementedError`; exploration is epsilon-greedy. ContinualBench's
   Sawyer arm is continuous (`continual_bench/envs/mujoco/sawyer_xyz_env.py:152` → `Box`).
   **This is the blocker for anything running at all.**
2. `models/rssm.py rollout_imagination` computes
   `action_dist.log_prob(torch.round(action.detach()))`. The `round` is a one-hot artifact and is
   nonsense for continuous actions — it will silently produce garbage or NaN. Unused under
   `actor_grad: 'dynamics'`, but still *computed*.
3. `utils/buffer.py TransitionBuffer._sample_idx` rejects only chunks straddling the write
   pointer — **sampled sequences can span two episodes.** Compensated by `nonterms` masking
   inside `rssm_imagine` (DreamerV1-style reset), weaker than PEG's episode chunking +
   `prioritize_ends`.
4. Single-env only (`collect_seed_episodes(env)`); our `src/train.py` uses `SubprocVecEnv` with
   `cfg.num_envs`.
5. Config is a per-env `@dataclass` (`training/config.py`); ours is YAML under `cfgs/algorithms/`.
6. `Trainer` owns its own loop and is wandb-coupled; we need `BasePolicy` + `src/logger.py`.
7. No Plan2Explore / ensemble (PEG needs it). No goal conditioning (LEXA layer).
8. Minor: `_optim_initialize` puts `self.DiscountModel` in `world_list` unconditionally, but it
   is only created when `config.discount['use']`. Crashes if discount is disabled.
9. Minor: `config.obs_dtype = np.uint8` default (MinAtar pixels) → must be `np.float32`.

### Current repo state

- `src/algorithms/dreamerv2.py` is a stalled half-port of `peg/dreamerv2/common/nets.py`:
  imports `torch.nn` but every body is still TF (`tf.zeros`, `@tf.function`,
  `prec.global_policy()`). It cannot run. **Delete it** — superseded by this plan.
- `src/buffer.py:60 ReplayBuffer` is flat transition-based with uniform index sampling. Kept for
  SAC; Dreamer gets its own sequence buffer.

---

## Vendoring decision

`dreamerv2/` is a clone with its own `.git` and is not a submodule. Do **not** develop in place —
we are diverging substantially and want our changes in CREMMA's history.

Copy the package into our tree and treat the clone as read-only reference:

```
src/algorithms/dreamer/
  __init__.py
  rssm.py          # from models/rssm.py + utils/rssm.py (merged)
  dense.py         # from models/dense.py, unchanged
  actor.py         # REWRITTEN: ContinuousActionModel
  algorithm.py     # from utils/algorithm.py (compute_return), unchanged
  module.py        # from utils/module.py (FreezeParameters), unchanged
  buffer.py        # Phase 1: episode-aware sequence buffer
  dreamerv2.py     # DreamerV2(BasePolicy) — from training/trainer.py, restructured
  gc_dreamer.py    # Phase 5
  goal_picker.py   # Phase 6
  morefree.py      # Phase 7
cfgs/algorithms/dreamerv2.yaml
```

Drop entirely: `models/pixel.py` (CNN), `training/config.py` (→ YAML),
`training/evaluator.py`, `utils/wrapper.py` (MinAtar/gym wrappers), `test/`.
Keep `dreamerv2/` in the tree as reference; add it to `.gitignore` or leave untracked.

---

## Phase 1 — Continuous actor + minimal wiring  ← **start here**

The shortest path to "it runs on ContinualBench at all."

1. Copy the files listed above into `src/algorithms/dreamer/`; fix imports.
2. Write `ContinuousActionModel` in `actor.py`, replacing `DiscreteActionModel`:
   - Same MLP trunk (`deter_size + stoch_size` → `node_size` × `layers`), output `2 * action_size`.
   - `dist: 'trunc_normal'` (DreamerV2's continuous default) or `tanh_normal`. Apply
     `mean_scale`/`init_std` from the config dict, which is already plumbed through.
   - `forward()` must return `(action, action_dist)` — reparameterized `rsample()` so gradients
     flow for `actor_grad: 'dynamics'`.
   - `add_exploration()`: Gaussian noise clipped to the action bounds, not epsilon-greedy.
3. Fix `rollout_imagination`'s `torch.round` — drop it on the continuous path.
4. Set `pixel: False`, `obs_dtype: np.float32`, `actor_grad: 'dynamics'`,
   `rssm_type: 'continuous'`, `rssm_info: {deter_size: 200, stoch_size: 50, min_std: 0.1}`.
5. Fix the `DiscountModel`/`world_list` ordering bug (finding 8).

**Test:** `representation_loss` and `actorcritc_loss` run end-to-end on random ContinualBench
rollouts without shape errors; losses are finite.

## Phase 2 — Episode-aware sequence buffer

Replace `TransitionBuffer` with a buffer matching PEG's `common/replay.py` semantics
(`replay: {minlen: 50, maxlen: 50, prioritize_ends: True}`, `dataset: {batch: 16, length: 50}`):
- Store complete episodes; never sample a chunk spanning an episode boundary.
- `prioritize_ends: True` — bias chunk start toward episode end.
- Emit `is_first` alongside `nonterms`.
- Actions time-aligned: `action[t]` is the action that *produced* `obs[t]`. Note the base repo
  handles this in `_shift_sequences`; preserve that convention exactly and assert it.
- One in-progress episode buffer **per env index** (we use `SubprocVecEnv`), committed on `done`.

**Test:** no chunk crosses a boundary; `is_first` set exactly at episode starts; action alignment.

## Phase 3 — `DreamerV2(BasePolicy)` + CREMMA integration

Restructure `Trainer` into an agent implementing `predict/update/save/load`:
- `predict(obs, deterministic)` — `BasePolicy` has nowhere to pass recurrent state, so the agent
  **owns its RSSM state internally**, one per env index. `src/train.py` must call a new
  `reset_state(env_idx)` on done.
- `update(replay_buffer, logger, step)` — wraps `train_batch`, logs via `src/logger.py` (drop wandb).
- `cfgs/algorithms/dreamerv2.yaml` mirroring the SAC config style; replace `training/config.py`.
- Register in `src/train.py:155 _build_policy`.

**Milestone: DreamerV2 trains on a single ContinualBench task and beats random.**
Do not proceed past this point until it holds.

## Phase 4 — Plan2Explore (ensemble disagreement)

Port `peg/dreamerv2/expl.py`. Adds an ensemble of one-step latent predictors; variance across
the ensemble is the intrinsic reward. Needed as PEG's exploration-value signal. This is where
`ensemble > 1` re-enters — the base repo has no ensemble at all.

## Phase 5 — Goal conditioning (LEXA layer)

Port `peg/dreamerv2/gc_agent.py` → `gc_dreamer.py`:
- `gc_input: 'embed'` | `'state'` — goal encoded and concatenated to the model state before
  actor/critic. **Thread an optional `goal` argument through actor/critic from Phase 1 onward**
  so this is an extension, not a rewrite.
- `gc_reward: 'dynamical_distance'` — learned temporal-distance head trained on replay pairs,
  used as the goal-reaching reward. Port the DD head and its optimizer.
- Two behaviors: goal-conditioned policy + exploration policy (Phase 4).
- `obs2goal` — maps observation to goal subspace. `src/train.py:70 _compute_goal_vector` /
  `_augment_obs` already does goal augmentation; wire `obs2goal` to that convention.

## Phase 6 — PEG

Port `peg/dreamerv2/goal_picker.py` (826 lines) → `goal_picker.py`:
- Go-Explore loop: run the GC policy toward a chosen goal, hand off to the exploration policy for
  the remainder (`goal_policy_rollout_percentage: 0.3–0.5`, `subgoal_threshold`).
- `goal_strategy: 'SubgoalPlanner'` — CEM shooting in latent space over candidate goals, scoring
  each by imagined exploration value (`cost_use_p2e_value: True`,
  `planner: {horizon: 25, batch: 500, cem_elite_ratio: 0.2, optimization_steps: 5}`).
- Implement cheaper baselines first — `SampleReplay`, then MEGA/SkewFit — as correctness
  references for the planner.

## Phase 7 — MoReFree

Prerequisite: ContinualBench made **autonomous / reset-free** (no `env.reset()` between episodes).
Port the PEG→MoReFree delta:
- `train_env_goal_percent: 0.2` — mix task-relevant goals (env goals + init states) into the goals
  used for imagination training. This is "prioritizing task-relevant states."
- `sample_env_init` — sample initial states as goals, so the agent learns to reset itself.
- `pseudo_goal_sampling` — split the mixed budget half env-goals / half env-inits.
- `alpha_s`, `alpha_e`, `alpha_decay_rate` — scheduled exploration ratio.
- Port `resetfree/env.py` + `goal_picker_wrapper.py` semantics onto ContinualBench.
- Reset-free replay: `MoReFree/dreamerv2/common/cr_replay.py` (590 lines) diverges from PEG's
  `replay.py` — revisit Phase 2 with it in hand.

---

## Ordering summary

1. Continuous actor + wiring  ← **blocker for everything**
2. Episode-aware sequence buffer
3. `DreamerV2(BasePolicy)` + CREMMA integration  ← **first real milestone**
4. Plan2Explore ensemble
5. Goal conditioning (LEXA)
6. PEG goal picker
7. MoReFree + autonomous ContinualBench

Cleanup before starting: `rm src/algorithms/dreamerv2.py`.
