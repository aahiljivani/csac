# CB_plan: Porting ContinualBench to the GPU (JAX + MJX / MuJoCo Warp)

This is an assessment of whether ContinualBench (`continual_bench/`) can be moved from CPU
`SubprocVecEnv` workers onto the GPU, which parts will hurt, which will be easy, and how to
approach the port if you have never built your own simulation environment before.

---

## TL;DR

**Yes, it is possible, and the environment is simpler than it first looks.** Most of the work is
mechanical. The hard part is the physics model (the XML), not the Python.

- **The env has no randomness.** Every "random" init space in the code is a box with
  `low == high`, so every reset of a given task produces exactly the same state. On the GPU,
  reset can be a plain copy of a precomputed snapshot. This removes the most expensive part
  of the current reset (250 physics steps to settle the arm).
- **The rewards are already written in vectorized form.** `reward_fns.py` and
  `torch_reward_utils.py` contain batched PyTorch versions of all six rewards. Translating
  `torch.where` → `jnp.where` gets you about 80% of the reward port.
- **The physics model does not load in MJX as-is.** I tested it: MJX (the JAX backend)
  rejects the scene with `NotImplementedError: (CYLINDER, MESH) collisions not implemented`.
  After I patched cylinders into boxes and turned off collision on the two dense gripper meshes,
  it loaded and stepped.
- **Contacts are the real GPU bottleneck, as you suspected.** The patched model allocates
  **4,525 contact slots per environment per step**, even though at most **49** contacts are ever
  active. Every slot is computed on every step, in every env, so most of the work is wasted.
  This is the "too many dynamic object collisions" problem, measured.
- **Recommended path:** write the env once against the `mujoco.mjx` API. Use
  `impl="jax"` to develop and debug on your Mac's CPU, then switch to `impl="warp"`
  (MuJoCo Warp) on an NVIDIA GPU. MuJoCo 3.15, which you already have installed, exposes
  `mjx.put_model(m, impl="warp")` and `mjx.put_data(..., naconmax=..., njmax=...)`, so the
  switch is one argument, not a rewrite. MuJoCo Warp only allocates the contacts you budget
  for (`naconmax`), which addresses the 4,525-slot problem directly.
- **Keep SAC in PyTorch to start with.** Pass GPU arrays between JAX and PyTorch with DLPack
  (zero-copy) instead of porting the agent too.

> **A note on terms.** "JAX" is the library (`jax`, `jax.numpy as jnp`). "MJX" is MuJoCo's JAX
> API. "MuJoCo Warp" (MJWarp, `mujoco_warp`) is a separate GPU implementation of MuJoCo written
> in NVIDIA Warp. Since MuJoCo 3.3.x you reach it *through* the MJX API with `impl="warp"`.

---

## Part 0: What I actually tested (evidence)

All numbers below were measured on this repo's real `sawyer_xyz/sawyer_bench.xml`, with
`mujoco 3.15.0`, `mujoco-mjx 3.15.0` and `jax 0.6.2`, on your Mac's **CPU** (no GPU here).
The scripts lived in a scratch folder and nothing in the repo was changed.

| Check | Result |
|:------|:-------|
| Model size | `nq=27`, `nv=25`, `nu=2`, 51 bodies, 117 geoms (**63 can collide**), 1 mocap body, 1 weld equality |
| Joints | 9 hinge, 4 slide, **2 free** (peg and block) |
| Solver options | `timestep=0.0025`, **Newton**, **50 iterations**, `tolerance=1e-10`, **elliptic** cone, dense Jacobian |
| Colliding geom types | 39 box, 15 cylinder, 4 capsule, 2 sphere, 1 plane, **2 mesh (3,021 and 6,693 vertices)** on the wrist and gripper base |
| Collision filtering | No `<exclude>` and no explicit `<pair>`; all colliding geoms use `contype=1 conaffinity=1` (everything can hit everything) |
| Candidate geom pairs (before broadphase) | **1,761** |
| Native C MuJoCo step | about **53 µs** per physics step; max **49** active contacts in a random rollout |
| `mjx.put_model` on the unmodified model | **Fails**: `(CYLINDER, MESH) collisions not implemented` |
| Same, with pyramidal cone | **Fails** with the same error, so the cone is not the blocker |
| Same, with dense meshes non-colliding | **Fails**: `(CYLINDER, BOX) collisions not implemented` |
| Cylinders → boxes and dense meshes non-colliding (cone left elliptic) | **Loads and steps** |
| Contact slots MJX allocates for that patched model | **4,525 per env** |
| MJX on CPU, 1 env | 9 s to compile, then **7.2 ms/step** (about 140× slower than native C) |
| MJX on CPU, `vmap` of 64 envs | 13 s to compile, then **782 ms** per batched step |
| MJX vs native after 300 steps of pushing the hand into the table | max qpos difference **0.25**, no NaNs |
| MuJoCo Warp | **Not tested.** The install ran out of disk space: your startup disk is full (about 239 MB free). Its claims below come from its documentation. |

How to read the CPU timings: MJX is *meant* to be slow on CPU, so these numbers are not a
verdict on GPU speed. They show that the model is heavy for MJX, and they point to the
contact count as the reason. On a GPU you want the per-env work to be small enough that
thousands of envs fit in one batch. With 4,525 contact slots, it isn't.

The 0.25 qpos difference is expected: float32 vs float64 arithmetic, a different contact
pipeline and a chaotic contact-rich rollout. It means you **cannot** check the port by
comparing trajectories bit for bit over long horizons. Part 5 explains how to validate instead.

---

## Part 1: A beginner's mental model

### What an "environment" really is

Strip away the gym wrappers and an RL environment is two functions:

```
reset(task)              -> state, obs
step(state, action)      -> state', obs', reward, done, info
```

`state` is everything that changes over time. In ContinualBench that is MuJoCo's `MjData`
(`qpos`, `qvel`, `mocap_pos`, …) plus a few Python attributes the env keeps on `self`:
`curr_path_length`, `_prev_gripper`, `init_tcp`, `init_data` and `obj_init_pos`.

`model` is everything that **doesn't** change: the XML compiled into `MjModel` (masses,
geometry, joint limits, solver options).

### Why the CPU version is shaped the way it is

Today, each env is a Python **object** that owns its own `MjData` and mutates it in place
(`self.data.mocap_pos = ...`, `self._prev_gripper = ...`). `SubprocVecEnv` runs N of these objects
in N processes and sends actions and observations over pipes. Your 10 CPU workers each do
about 53 µs of physics plus a lot of Python overhead per step.

### What changes on the GPU

A GPU is fast when it does **the same arithmetic on thousands of things at once**. So you stop
thinking "N env objects" and start thinking "**one** env function, applied to a **batch** of
states". In JAX that is:

```python
step_batch = jax.jit(jax.vmap(step, in_axes=(None, 0, 0)))   # model shared, state/action batched
```

For this to work, `step` has to follow JAX's rules. These rules are most of what is new for you:

1. **Pure functions.** No `self.x = ...`. Everything that changes goes into `state` and comes
   back out. Write `new_state = step(state, action)`.
2. **No Python `if` on array values.** `if tcp_to_obj <= 0.03:` breaks under `jit`. Use
   `jnp.where(cond, a, b)`, which computes both branches and picks one per element. For whole
   code paths, use `jax.lax.cond` or `jax.lax.switch`.
3. **Fixed shapes.** Array sizes can't depend on data. (This is why MJX pre-allocates 4,525
   contact slots: it can't grow the list at runtime.)
4. **Immutable arrays.** `x[3] = 1.0` becomes `x = x.at[3].set(1.0)`.
5. **Explicit randomness.** No `np.random`. Pass a `jax.random.PRNGKey` in and out.
   (ContinualBench has no real randomness, so you barely need this.)
6. **Loops become `lax.fori_loop` or `lax.scan`.** Python `for` loops get unrolled at compile
   time. That's fine for 5 iterations, bad for 500.
7. **Look up names once, outside `jit`.** `mj_name2id(model, ..., "leftpad")` is a Python call.
   Do it when you build the env, store the integer ids and index arrays with them inside `step`.

### NumPy → `jnp` cheat sheet for this codebase

| CPU code here | GPU (JAX) version |
|:--------------|:------------------|
| `np.linalg.norm(a - b)` | `jnp.linalg.norm(a - b, axis=-1)` |
| `np.clip(x, lo, hi)` | `jnp.clip(x, lo, hi)` |
| `np.hstack((...))` | `jnp.concatenate([...], axis=-1)` |
| `if c: r = 10.0` | `r = jnp.where(c, 10.0, r)` |
| `float(x <= 0.024)` | `(x <= 0.024).astype(jnp.float32)` |
| `min(max(0, a), e)` | `jnp.clip(a, 0, e)` |
| `self.data.body("hand").xpos` | `d.xpos[HAND_BODY_ID]` (id looked up once at build time) |
| `self.data.site("x").xpos` / `_get_site_pos("x")` | `d.site_xpos[SITE_ID]` |
| `self.data.geom("handle").xpos` | `d.geom_xpos[GEOM_ID]` |
| `self.data.joint("doorjoint").qpos` | `d.qpos[DOOR_QPOS_ADR]` (from `m.jnt_qposadr`) |
| `qpos[20:23] = p` | `qpos = qpos.at[20:23].set(p)` |
| `mujoco.mj_step(m, d, nstep=5)` | `lax.fori_loop(0, 5, lambda _, d: mjx.step(mx, d), d)` |
| `mujoco.mj_forward(m, d)` | `mjx.forward(mx, d)` |
| `self.data.mocap_pos = p` | `d = d.replace(mocap_pos=p)` |
| `torch.where(c, a, b)` (in `reward_fns.py`) | `jnp.where(c, a, b)` |

---

## Part 2: File-by-file walkthrough

Order: from the outside (package entry) to the inside (physics and rewards). Each file gets
**what it does**, **what is CPU-specific about it**, and **what to do on the GPU**.

### `continual_bench/__init__.py`, `continual_bench/__about__.py`

**Does:** Package metadata and a license header.
**GPU:** Nothing. Add a new module (for example `continual_bench/mjx_env.py`) next to the existing code
so the CPU env keeps working as the reference implementation. **Don't delete the CPU env.**
You need it for validation (Part 5) and for rendering videos.

### `continual_bench/envs/__init__.py`

**Does:** Builds the public `ContinualBenchEnv` class at runtime with
`type("ContinualBench", (SawyerBenchEnv,), {"__init__": initialize})`. `initialize`:
1. saves the **global** `np.random` state and seeds it,
2. calls `SawyerBenchEnv.__init__`, which loads the XML, builds `MjModel`/`MjData` and calls `set_task("button")`,
3. calls `env.seed(seed)` to seed the gym spaces,
4. does one `reset()` with `_freeze_rand_vec=False`, then freezes it,
5. restores the global RNG.

**CPU-specific:** Global RNG juggling and gym `Box.sample()`. Every sampled `Box` here has
`low == high` (`hand_init_spaces`, `_random_reset_space`), so the seed changes **nothing about
the env**. It only matters for the policy.
**GPU:** Replace this with a plain factory, for example `make_mjx_env(impl="jax"|"warp") -> (env_params, reset_fn, step_fn)`.
There's no RNG to thread through. Write that down in a comment so nobody adds one by accident.

### `continual_bench/envs/asset_path_utils.py`

**Does:** Joins file names onto the `assets/` directory.
**GPU:** Unchanged. You still load the XML from disk with `mujoco.MjModel.from_xml_path` (or
`mujoco.MjSpec.from_file` if you patch it in code) on the CPU, then hand it to MJX.

### `continual_bench/envs/assets/` (the XML scene): **the main pain point**

**Does:**
- `sawyer_xyz/sawyer_bench.xml` is the top-level scene. It includes the table/lights scene, the
  Sawyer robot (`objects/assets/xyz_base.xml`) and all six objects (button box, door, window,
  faucet, peg + peg box, block). It also declares 2 position actuators (the fingers,
  `kp=400`) and one **weld equality** between the `mocap` body and the `hand`. That weld is how
  the arm is driven: you move the invisible mocap body and the weld drags the hand along.
- `scene/basic_scene.xml` sets the **solver options**: `timestep 0.0025`, `Newton`,
  `iterations 50`, `tolerance 1e-10`, `cone elliptic`, `jacobian dense`. It also defines the
  table, which is a **cylinder** of radius 1.025 (a colliding cylinder, which MJX can't pair with
  boxes).
- `objects/assets/*.xml` hold the object geometry. Most objects are built from many small
  colliding boxes and cylinders (the window alone has 17 boxes, 2 cylinders and 3 capsules).
  `objects/meshes/*.stl` are the meshes, and most are visual-only (`contype=0`).

**Why it's hard on the GPU:**

1. **Unsupported collision pairs in MJX-JAX.** 15 colliding cylinders (table, button, door handle,
   window, faucet, plug) and 2 dense meshes give `CYLINDER×BOX` and `CYLINDER×MESH` pairs,
   which MJX-JAX doesn't implement. (MuJoCo Warp has general convex collision and, per its
   docs, supports these pairs. Not verified here, see Part 0.)
2. **Dense collision meshes.** The wrist (`right_l6`, 3,021 vertices) and gripper base
   (`right_hand`, 6,693 vertices) have colliding meshes. Convex mesh collision cost grows with
   vertex count. MJX's docs recommend a few hundred vertices at most.
3. **No collision filtering.** Every colliding geom has `contype=1 conaffinity=1`, so the
   button can "collide" with the faucet, the window with the door and so on. Those pairs are
   physically far apart and never touch, but MJX still evaluates them and reserves slots for
   them. That's how you get **1,761 candidate pairs → 4,525 contact slots**.
4. **Expensive solver settings.** Newton with 50 iterations at `1e-10` is tuned for CPU
   accuracy. On a GPU, every env in the batch runs as many iterations as the slowest one.

**What to do** (in this order, re-measuring after each step):

1. **Make a GPU copy of the scene**, for example `sawyer_xyz/sawyer_bench_mjx.xml`, and leave the
   original alone. You can also patch in code with `mujoco.MjSpec` (that's what my probe did),
   but a separate XML file is easier to diff and review.
2. **Collision filtering with bitmasks.** This is the single biggest win. Give each group its own bit:
   - gripper fingers and pads: collide with everything,
   - table: collides with the gripper, the block, the peg and the arm,
   - each static object (button box, door, window, faucet, peg box): collides **only** with
     the gripper, plus its own moving part where needed (the door with its frame, the peg with the peg box),
   - block and peg (the free bodies): collide with the table, the gripper and the peg box.

   Two geoms collide if `(contype_a & conaffinity_b) || (contype_b & conaffinity_a)`. Write
   the matrix out on paper first. Add `<exclude body1=... body2=.../>` for any leftovers. Goal:
   go from 1,761 candidate pairs to a few hundred. Check the result with
   `mjx.put_data(...).contact.dist.shape` (MJX-JAX) or by watching `ncon` (MuJoCo Warp).
3. **Replace or simplify the dense meshes.** The wrist and gripper-base meshes only need to stop
   the arm passing through things. Swap them for 1–3 capsules or boxes, or set
   `maxhullvert="64"` on the mesh asset. Keep the visual mesh, and only change the *collision* geom.
4. **Cylinders → capsules or boxes, for MJX-JAX only.** I measured the boxes version loading. For
   the table, a thin **box** is the natural choice. For round handles, a capsule is closer to the
   original shape. If you go straight to MuJoCo Warp, try leaving the cylinders in place first.
5. **Solver budget.** Try `iterations="4"` to `"10"` with `ls_iterations="4"`–`"8"`, and a looser
   tolerance. Check that tasks can still be solved with a scripted policy (Part 5). Elliptic cone loaded fine
   in MJX after the patch. If it's slow or unstable, `cone="pyramidal"` is the usual GPU choice.
6. **Re-check every reward constant afterwards.** Rewards depend on geometry: thresholds
   like `obj_to_target <= 0.024` for the button and site offsets like `+0.193` on the button's z.
   If swapping a cylinder for a box moves a contact surface, a threshold may need adjusting. Do this
   one object at a time.

This is also where your earlier "too many dynamic object collisions" experience fits: the
problem is real, but most of the colliding pairs are between objects that never touch.
Filtering them out is cheap and doesn't change the task.

### `continual_bench/envs/mujoco/mujoco_env.py`

**Does:** A copy of gym's MuJoCo base classes.
- `BaseMujocoEnv`: resolves the XML path, creates `MjModel`/`MjData`, stores `init_qpos`/`init_qvel`,
  checks that `render_fps == 1/dt` (80 Hz = 1 / (0.0025 × 5)), and provides `do_simulation`, `set_state` and `dt`.
- `MuJocoPyEnv`: the legacy `mujoco_py` backend. **Unused** (`mujoco_py` import is wrapped in `try`).
- `MujocoEnv`: the real backend. `_step_mujoco_simulation` writes `ctrl`, calls
  `mj_step(nstep=frame_skip)` and then `mj_rnePostConstraint`. It also implements `get_body_com`
  (which returns body **xpos**, not the actual COM) and offscreen rendering with `mujoco.Renderer`.
- `_assert_task_is_set`: a decorator that raises an error if `set_task` was never called.

**GPU:**

| Piece | Action |
|:------|:-------|
| `_initialize_simulation` | Keep it on CPU: `MjModel.from_xml_path` → `mjx.put_model(m, impl=...)` once. |
| `_reset_simulation` (`mj_resetData`) | Gone. Reset becomes "load the precomputed snapshot" (see `sawyer_bench.py`). |
| `set_state` | `d.replace(qpos=..., qvel=...)` then `mjx.forward`. |
| `_step_mujoco_simulation` | `d = d.replace(ctrl=ctrl)` then `lax.fori_loop(0, 5, ...)` over `mjx.step`. |
| `mj_rnePostConstraint` | **Drop it.** It fills `cacc`/`cfrc_*`, which no observation or reward reads. |
| `get_body_com(name)` | `d.xpos[body_id]`. Keep the name: it's body **origin** position, not center of mass. |
| `render` | **Stays on CPU.** MJX doesn't render. For eval videos, take one env out of the batch with `mjx.get_data(m, d_one)` and render it with the existing `mujoco.Renderer`. Eval videos are rare, so this is fine. |
| `MuJocoPyEnv` | Ignore it. |

### `continual_bench/envs/mujoco/sawyer_xyz_env.py`

**Does:** Sawyer-specific base logic (inherited from Meta-World).
- `SawyerMocapBase`: `get_endeff_pos` (= `hand` body xpos), `tcp_center` (the midpoint of the
  `rightEndEffector` and `leftEndEffector` sites), pickling helpers, and `reset_mocap_welds`, which sets the weld's
  relative pose to identity once at construction.
- `SawyerXYZEnv`:
  - `set_xyz_action`: clips the action to [-1, 1], scales it by `action_scale = 1/100`, adds it to
    `mocap_pos`, **clips mocap_pos to `hand_low`/`hand_high`** (`[-0.8,-0.8,0.05]`–`[0.8,0.8,0.5]`),
    and pins `mocap_quat = [1, 0, 1, 0]`.
  - `step`: calls `set_xyz_action`, then runs physics with `ctrl = [a3, -a3]` for 5 substeps, then
    `_get_obs`, then **clips obs to `sawyer_observation_space`**, then `evaluate_state`, then
    `done = (t == 500) or success`.
  - `_reset_hand`: **50 iterations × 5 substeps = 250 physics steps** with the mocap held at
    `hand_init_pos` and the fingers open, then records `init_tcp`.
  - `_gripper_caging_reward`: the shared grasp-shaping reward (used by peg and block).
  - Meta-World leftovers that this benchmark never uses: `set_task(task)` with pickle,
    `_get_curr_obs_combined_no_goal`, the 39-dim `_get_obs`, `_get_obs_dict`,
    `touching_object`, `_get_state_rand_vec`, `discrete_goal_space` and `reset()` (see the note below).

> **Note:** `SawyerBenchEnv.reset()` **does not** call `SawyerXYZEnv.reset()`. It calls
> `_reset_simulation()` and `reset_model()` directly, so the frame-stacking code in
> `SawyerXYZEnv.reset` is dead. Also, `_did_see_sim_exception` is never set to `True`, so the
> "instability" branch in `step` is dead too.

**GPU:**

```python
def set_xyz_action(d, a):                      # a: (4,)
    a = jnp.clip(a, -1.0, 1.0)
    pos = jnp.clip(d.mocap_pos[0] + a[:3] * ACTION_SCALE, HAND_LOW, HAND_HIGH)
    return d.replace(mocap_pos=pos[None], mocap_quat=jnp.array([[1., 0., 1., 0.]]))

def physics_step(mx, d, a):
    d = set_xyz_action(d, a)
    d = d.replace(ctrl=jnp.array([a[3], -a[3]]))
    return jax.lax.fori_loop(0, FRAME_SKIP, lambda _, d: mjx.step(mx, d), d)

def tcp_center(d):
    return 0.5 * (d.site_xpos[RIGHT_EE_SITE] + d.site_xpos[LEFT_EE_SITE])
```

- `_reset_hand` is **the most expensive part of reset** (250 physics steps), and its result is
  **identical every time** for a given task. Run it **once per task on the CPU with native
  MuJoCo** at env build time and store `qpos`, `qvel`, `mocap_pos`, `mocap_quat` and `init_tcp`. That's
  6 small snapshots. GPU reset is then a gather from a `(6, nq)` array. Easy and a big win.
- `_gripper_caging_reward` → port the **batched** version already written in
  `torch_reward_utils.gripper_caging_reward` (see below), and pass in `obj_init_pos`, `init_tcp`
  and the pad positions explicitly instead of reading them from `self`.
- `max_path_length` and the "step after truncation raises an error" check: replace them with a `t` counter in state,
  plus auto-reset (Part 3).
- Everything listed above as a Meta-World leftover: don't port it.

### `continual_bench/envs/mujoco/sawyer_bench.py` (the main env, 916 lines)

**Does:** Everything specific to ContinualBench.

**Static data:** `all_tasks`, `hand_init_spaces` (degenerate `Box`es), `obj_init_positions`.

**`set_task(name)`:** Stores a `TaskSpec` (name, hand init pos, object init pos) and the camera name.
Doesn't touch physics.

**`reset_model()`:** Runs on every reset.
1. Calls `_reset_hand()` (250 steps, see above).
2. For **every** object (not just the active task's), it sets the body position in **`self.model`**
   (`model.body("button1").pos`, `"door"`, `"window"`, `"faucet"`, `"pegBox"`) and goal site
   positions (`doorGoal`, `faucetGoal`, `pegGoal`). It sets the window's slide joint to 0.2, places the
   peg (`qpos[13:20]`) and block (`qpos[20:23]`), and fills `self.init_data[task]` with per-task
   constants (`target_pos`, `obj_to_target_init`, `window_handle_pos_init`, `midpoint`, …).
3. Calls `mj_forward` and returns `_get_obs()`.

**`_get_obs()`:** Builds the 26-D observation:

| idx | content | source |
|:---:|:--------|:-------|
| 0:3 | hand pos | `xpos[hand]` |
| 3 | gripper openness | `‖xpos[rightclaw] − xpos[leftclaw]‖ / 0.1`, clipped to [0, 1] |
| 4:7 | button pos | `xpos[button] + [0, 0, 0.193]` |
| 7:10 | door handle pos | `geom_xpos[handle]` |
| 10 | door angle | `qpos[doorjoint]` |
| 11:14 | window handle | `site_xpos[windowHandleCloseStart]` |
| 14:17 | faucet handle | `site_xpos[faucetHandleStartClose] + [0, 0, −0.01]` |
| 17:20 | peg end | `site_xpos[pegEnd]` |
| 20:23 | block | `xpos[block]` |
| 23 | gripper direction | openness − `_prev_gripper` (**hidden state carried on `self`**) |
| 24, 25 | pad y offsets | `xpos[leftpad].y − hand.y`, `xpos[rightpad].y − hand.y` |

**`sawyer_observation_space`:** A `Box` used to clip obs. (It's rebuilt from scratch on **every
step**, which is a small CPU-side waste on its own.)

**`evaluate_state(obs, action)`:** Computes **all six** rewards and info dicts every step and
returns only the active task's reward.

**`*_compute_reward`:** The per-task shaped rewards. `block_compute_reward` also computes **four**
extra debug versions of the grasp reward through PyTorch (`debug_own_grasp_fn`,
`debug_grasp_reward_tcp`, …) just to log differences.

**`get_init_state()`:** Packages `init_data` + `init_tcp` with a leading batch axis for the
model-based reward functions in `reward_fns.py`.

**What's easy on the GPU:**

- **All of `reset_model`'s model edits are constants.** `obj_init_positions` never change, so
  bake the body and site positions into the GPU XML (or set them once on the `MjModel` before
  `mjx.put_model`). With MJX the `Model` is shared by the whole batch, so per-env model edits
  would be awkward. Here, you don't need any.
- **`init_data` is a constant per task.** Compute it once on the CPU after the per-task hand
  reset, then store it as small `jnp` arrays (stack per-task fields into `(6, ...)` arrays and
  index them with `task_id`).
- **The observation** is about 15 lines of indexing (table above), with ids looked up once.
  `_prev_gripper` becomes a field in your env state.
- **Rewards** are short, branch-light math. See `reward_fns.py` below for the batched template.

**What will bite you:**

1. **A latent bug: `handleCloseStart` doesn't exist.** `reset_model` does
   `self._get_site_pos("handleCloseStart")`, but the scene has `windowHandleCloseStart` and no
   `handleCloseStart`. `mj_name2id` returns **−1** and `site_xpos[-1]` silently reads the **last**
   site, which is the peg box's `hole`. So `window_handle_pos_init` (used for the window
   reward's `in_place` and `reach` margins) is the hole position + 0.2 in y. Your results so far
   were trained with this. **Decide on purpose:** copy it exactly for the parity port (use the
   `hole` site id), then fix it as a separate, labelled change.
2. **Another −1 lookup:** `_set_pos_site("button1", ...)` writes to `site_xpos[-1]` too (there's
   no site called `button1`; it's a body). It's harmless because the next `mj_forward` overwrites
   it, but don't port it. **Lesson:** in your GPU code, wrap name lookups in a helper that
   **asserts id ≥ 0**.
3. **Hard-coded qpos/qvel indices.** `qpos[13:20]` (peg) and `qpos[20:27]` (block) are right, but
   `_qvel[20:26] = 0` for the block is **off by one**: the block's DOFs start at `dofadr 19`.
   It's harmless today because `mj_resetData` already zeroed `qvel`, but it shows why you should
   use `m.jnt_qposadr[jid]` / `m.jnt_dofadr[jid]` instead of literals.
4. **`self.obj_init_pos` leaks between tasks.** `_gripper_caging_reward` reads
   `self.obj_init_pos`, which is the **active task's** object init position. When training
   `button`, the block reward (computed but discarded) uses the button's position. In the GPU
   version, pass the *right* task's `obj_init_pos` explicitly. If you only compute the active
   task's reward, the problem goes away.
5. **The block reward mixes two TCP definitions.** It uses an observation-derived TCP
   (`hand − [0, 0, 0.045]` via `reward_fns._get_tcp_center`) and observation-derived pad y-positions,
   while other tasks use the true `tcp_center` from sites. To reproduce current behaviour, copy it
   exactly. Since `block` is the task that never learned, this is worth revisiting later. **Don't** fix
   it during the port, or you won't know whether the port or the fix changed results.
6. **Rewards read obs *after* clipping.** `step` clips the obs to the observation box before passing it
   to `evaluate_state`. Keep that order.
7. **Branching.** `if tcp_to_obj <= 0.03`, `if radius <= threshold`, `if hand[2] >= floor` and similar
   all become `jnp.where`. Watch `_door_reward_pos`: `0.04 * log(radius − threshold)` is evaluated on
   **both** branches by `jnp.where`, so it produces `nan` when `radius ≤ threshold`. Guard it:
   `jnp.log(jnp.maximum(radius - threshold, 1e-8))`. (`jnp.where` won't pick the nan here, but it can
   poison gradients if you ever differentiate through the reward, and it's a classic beginner trap.)
8. **`info` dicts.** Python dicts of floats per env become a fixed pytree of arrays, for example
   `{"success": (B,), "obj_to_target": (B,)}`. Only include what `src/train.py` and the logger read
   (`success`), plus a few for debugging.

**Only compute the active task.** Use `jax.lax.switch(task_id, [button_r, door_r, ...], d, obs, a)`.
All branches must return the same shapes. Since every env in a batch trains the same task in
CREMMA, you can also pick the function in Python outside `jit` and compile one step function per task
(6 compiles, simpler code). I recommend that at first.

### `continual_bench/envs/reward_utils.py`

**Does:** dm_control reward helpers. `tolerance` (1 inside bounds, sigmoid falloff outside),
`_sigmoids` (8 shapes, but only `long_tail` and `gaussian` are used), `hamacher_product`
(`ab / (a + b − ab)`), plus `inverse_tolerance` and `rect_prism_tolerance` (both unused).

**CPU-specific:** `hamacher_product` raises `ValueError` and `assert`s on its inputs, and
`tolerance` has Python `if margin == 0` and `if lower > upper` branches.
**GPU:** Port only what's used:

```python
def long_tail(x, value_at_1=0.1):
    scale = jnp.sqrt(1.0 / value_at_1 - 1.0)
    return 1.0 / ((x * scale) ** 2 + 1.0)

def gaussian(x, value_at_1=0.1):
    scale = jnp.sqrt(-2.0 * jnp.log(value_at_1))
    return jnp.exp(-0.5 * (x * scale) ** 2)

def tolerance(x, lower, upper, margin, sigmoid):          # sigmoid: Python function, chosen statically
    in_bounds = (lower <= x) & (x <= upper)
    d = jnp.where(x < lower, lower - x, x - upper) / jnp.maximum(margin, 1e-12)
    out = jnp.where(in_bounds, 1.0, sigmoid(d))
    return jnp.where(margin == 0, in_bounds.astype(x.dtype), out)

def hamacher_product(a, b):
    den = a + b - a * b
    return jnp.where(den > 0, a * b / jnp.where(den > 0, den, 1.0), 0.0)
```

Note the "double `where`" in `hamacher_product`: the inner one keeps the division finite on the
branch that gets thrown away. Use that pattern whenever a branch could divide by zero.
Drop the checks: you can't raise errors inside `jit`. Check ranges in your tests instead.

### `continual_bench/envs/torch_reward_utils.py` and `continual_bench/envs/reward_fns.py`

**Does:** **Batched PyTorch versions** of the reward helpers and all six task rewards. They were written for
the model-based agent from the original paper, which needs rewards computed from *predicted*
observations. They take `(B, obs_dim)` tensors plus `init_data` from `get_init_state()`, and use
`torch.where` instead of `if`. The tests (`tests/control_env.py`, `tests/keyboard_control.py`,
`tests/test_env.py`) use them. `src/` does not.

**GPU:** **This is your template.** Port `torch_reward_utils.py` → `jax_reward_utils.py` and
`reward_fns.py` → `jax_reward_fns.py` almost line by line (`torch.where` → `jnp.where`,
`torch.norm(..., dim=-1)` → `jnp.linalg.norm(..., axis=-1)`, `.clip` → `jnp.clip`).
Two differences from the env rewards to keep in mind:
- `reward_fns` gets TCP and pad positions **from the observation** (approximations), while the
  env's `*_compute_reward` uses **true** site and body positions for everything except `block`. In the GPU env
  you have the true `Data`, so pass the true values for button/door/window/faucet/peg and the
  obs-derived values for block, to match the current CPU env.
- Check every ported reward against `SawyerBenchEnv.*_compute_reward` (not against `reward_fns`)
  on recorded states. The env is the ground truth your agent was trained on.

### `tests/`

- `test_env.py`, `test_env_outputs.py` and `control_env.py` create the env, roll it out and compare the env reward
  with `reward_fns`. **Reuse the idea:** a parity test that runs CPU env and GPU env side by side
  (Part 5).
- `keyboard_control.py` is manual driving. It's useful for recording **scripted demo trajectories**
  that solve each task. You'll need those for validation.
- `env_output_diagnostic.txt` is a saved output dump and is fine as a reference.

### `src/train.py` (CREMMA side, where the GPU env plugs in)

**Does today:** For each task it builds a `SubprocVecEnv` of `ContinualBenchEnv`s wrapped in
`shimmy.GymV21CompatibilityV0`, then loops over 500 steps. Each step calls `agent.predict` (torch) and
`vec_env.step` (pipes to CPU workers), loops over `infos` in Python to get `success` and
`terminal_observation`, adds the 4-D goal vector, and does `num_envs` SAC updates per vector step.

**GPU:** The new env replaces `SubprocVecEnv` + shimmy + the Python per-env loops:

- `obs`, `reward`, `done`, `success` and `terminal_obs` come back as `(B, ...)` GPU arrays. No
  `infos` list, no `for idx, done in enumerate(dones)`.
- **Auto-reset inside the env:** after stepping, `state = where(done, reset_state[task], state)`
  and return the pre-reset obs as `terminal_obs` (you need it for the replay buffer, just like SB3's
  `terminal_observation`).
- **JAX ↔ PyTorch without copying:** `torch.utils.dlpack.from_dlpack(jax_array)` and
  `jax.dlpack.from_dlpack(torch_tensor)` share the same GPU memory. Keep SAC, the replay buffer and
  R&D in PyTorch.
- **Goal vector:** `_compute_goal_vector` builds a whole CPU env just to read `init_data[task].target_pos`.
  Precompute all six once and keep them on the GPU.
- **Update-to-data ratio:** `for _ in range(self.num_envs): agent.update(...)` means 10 updates per
  vector step today. With 1,024+ envs, that loop would do 1,024 updates per step. Make UTD an explicit
  config value. This is exactly what the parallel-SAC config and sweep setup from Arth Shukla's ManiSkill3
  post is designed to tune (many envs, fewer updates per env step, larger batches).
- **Videos and eval:** eval with a separate small batch. For videos, copy one env to CPU and render
  it with the existing `mujoco.Renderer` path.

---

## Part 3: The GPU env, end to end (sketch)

This is the shape to aim for. It isn't drop-in code, but each piece maps to a section above.

```python
# continual_bench/mjx_env.py
import jax, jax.numpy as jnp, mujoco, numpy as np
from mujoco import mjx
from flax import struct          # or chex.dataclass / a NamedTuple

TASKS = ["button", "door", "window", "faucet", "peg", "block"]
FRAME_SKIP, MAX_T = 5, 500

@struct.dataclass
class EnvState:
    data: mjx.Data        # physics
    t: jax.Array          # step counter            (was self.curr_path_length)
    prev_grip: jax.Array  # last gripper openness   (was self._prev_gripper)

def build(xml_path, impl="jax"):
    m = mujoco.MjModel.from_xml_path(xml_path)
    ids = lookup_ids(m)                     # every name → int, assert >= 0
    snapshots = precompute_resets(m, ids)   # native MuJoCo: per-task settled qpos/qvel/mocap,
                                            # init_tcp, init_data  → stacked (6, ...) arrays
    mx = mjx.put_model(m, impl=impl)
    return m, mx, ids, snapshots

def reset(mx, snaps, task_id):
    d = mjx.make_data(mx)
    d = d.replace(qpos=snaps.qpos[task_id], qvel=snaps.qvel[task_id],
                  mocap_pos=snaps.mocap_pos[task_id], mocap_quat=snaps.mocap_quat[task_id])
    d = mjx.forward(mx, d)
    grip = gripper_openness(d)
    s = EnvState(data=d, t=jnp.int32(0), prev_grip=grip)
    return s, get_obs(d, grip, grip)

def step(mx, snaps, task_id, reward_fn, s, a):
    d = physics_step(mx, s.data, a)                         # mocap + ctrl + 5 substeps
    grip = gripper_openness(d)
    obs = jnp.clip(get_obs(d, grip, s.prev_grip), OBS_LOW, OBS_HIGH)
    r, success = reward_fn(d, obs, a, snaps, task_id)
    t = s.t + 1
    done = (t >= MAX_T) | success
    s_new = EnvState(d, t, grip)
    s_reset, obs_reset = reset(mx, snaps, task_id)          # cheap: a copy plus a forward
    s_out = jax.tree_util.tree_map(lambda x, y: jnp.where(done, x, y), s_reset, s_new)
    obs_out = jnp.where(done, obs_reset, obs)
    return s_out, obs_out, r, done, success, obs            # last value = terminal obs

batched_step = jax.jit(jax.vmap(step, in_axes=(None, None, None, None, 0, 0)),
                       static_argnums=(3,))                 # reward_fn chosen per task
```

When you run with `impl="warp"`, also pass contact and constraint budgets to `mjx.make_data` /
`mjx.put_data` (`naconmax`, `njmax`). Start them a bit above what you measured natively (max 49
contacts in my random rollout), log overflow, and raise them if needed.

---

## Part 4: Pain points vs easy wins

### Easy (do these first, they're low risk)

| Item | Why it's easy |
|:-----|:--------------|
| Reset → snapshot copy | No randomness anywhere. Precompute 6 states once with native MuJoCo. Removes 250 substeps per reset. |
| `init_data`, goal vectors, model/site edits | All constants. Compute once and bake into the XML or arrays. |
| Observation | 26 indexed reads plus one carried scalar (`prev_grip`). |
| Rewards | A batched torch reference already exists (`reward_fns.py`). Mechanical `torch` → `jnp`. |
| Action → mocap | Three lines of `jnp.clip` and `replace`. |
| Frame skip | `lax.fori_loop(0, 5, ...)`. |
| Drop dead code | `mujoco_py`, Meta-World leftovers, `_did_see_sim_exception`, `mj_rnePostConstraint`, block debug rewards. |

### Medium

| Item | Why |
|:-----|:----|
| Auto-reset + terminal obs | You have to get the `jnp.where` over the whole state pytree right, and keep the pre-reset obs. |
| Training-loop rewrite | Removes Python per-env loops. Needs DLPack interop and a new UTD config. |
| Reproducing the quirks | The `handleCloseStart` → `hole` lookup, obs clipping before reward, the obs-based TCP for block. Each needs a test. |
| Rendering/eval videos | CPU round-trip for one env. Simple but separate code. |

### Hard (this is where the time goes)

| Item | Why |
|:-----|:----|
| **Collision model** | MJX-JAX can't handle cylinder × box/mesh pairs. 1,761 unfiltered pairs → 4,525 contact slots per env. Needs careful bitmask design and geometry swaps object by object. |
| **Physics parity** | float32, a different contact pipeline and fewer solver iterations all mean different dynamics. Tasks that depend on grasping (block, peg) or tight thresholds (button 0.024) are the most sensitive. |
| **Solver tuning** | Newton with 50 iterations at 1e-10 is too costly per env. Fewer iterations can make grasping slip. You need scripted-policy checks per task. |
| **The block task specifically** | It's a free body with a 0.01 kg mass and a huge `diaginertia` (1e5), a hand-tuned `solimp`/`solref` and `friction="2 0.1 0.002"`. Unusual contact parameters like these are the most likely thing to behave differently on GPU solvers. It's also the task that never learned, so be careful about blaming the port for it. |
| **MuJoCo Warp availability** | It needs an NVIDIA GPU with CUDA, and it couldn't be tested on this Mac. Plan your debugging on CPU with `impl="jax"`, and your speed work on a GPU machine. |

---

## Part 5: How to approach it (step by step)

Advice from people who have done this: **change one thing at a time, and never trust a step you haven't
measured.** Each phase ends with a check that has to pass before you move on.

**Phase 0: Learn the tools on a toy (1–2 days).**
Work through the official MJX tutorial notebook (`mjx/tutorial.ipynb` in the MuJoCo repo).
Write a 20-line script that `vmap`s `mjx.step` over a cartpole or a single falling box. Get
comfortable with `jit`, `vmap`, `lax.fori_loop`, `.replace(...)`, `jnp.where` and reading JAX error
messages. *Check: you can explain why `if x > 0:` fails under `jit`.*

**Phase 1: Record ground truth from the CPU env.**
With the **current** env, record for each task: (a) a scripted or keyboard demo that succeeds and
(b) some random-action rollouts. Save `qpos`, `qvel`, `mocap_pos`, action, obs, reward and success per step.
These are your tests. *Check: replaying the actions in the CPU env reproduces the saved values exactly.*

**Phase 2: Pure-function rewards and observations (no GPU physics yet).**
Port `reward_utils` / `reward_fns` / `get_obs` to `jnp`. Feed them the **CPU** env's `MjData`
values from Phase 1. *Check: obs and reward match the CPU env to about 1e-5 on every recorded
step, including the `hole`-site quirk for window.* This isolates the Python port from the physics.

**Phase 3: Get the scene to load and run in MJX on CPU.**
Make `sawyer_bench_mjx.xml`: cylinders → boxes/capsules, dense meshes → primitives or
`maxhullvert`, then collision bitmasks. Re-run the probe after each change and log candidate pairs and
contact slots. *Check: `mjx.put_model` succeeds and contact slots drop from 4,525 to a few hundred.*

**Phase 4: Physics parity, measured the right way.**
You can't expect identical trajectories (Part 0 showed a 0.25 qpos difference after 300 steps).
Instead check that: (a) **one-step** error from the same state is small (within about 1e-3), (b) the
Phase 1 **scripted demos still succeed** on every task, (c) contact counts are similar, and (d) nothing goes
NaN over long random rollouts. If a demo fails, the geometry or solver change in Phase 3 caused it.
Bisect.

**Phase 5: The env loop.**
Implement `reset` (snapshots), `step` (frame skip + obs + reward + auto-reset), `jit(vmap(...))`.
*Check: a batch of 8 envs on CPU replays the Phase 1 demos with the same success flags as Phase 4.*

**Phase 6: GPU and MuJoCo Warp.**
On an NVIDIA machine, install `mujoco-warp`, switch `impl="warp"` and set `naconmax`/`njmax`.
Try restoring the original cylinders here, since MuJoCo Warp may handle them. Measure steps/sec against
the number of envs (256, 1k, 4k, 8k). *Check: the Phase 4 demos still succeed, and throughput is well above
the current 10-worker `SubprocVecEnv` (measure the baseline first so you have something to compare against).*

**Phase 7: Plug into CREMMA.**
Write a `GPUContinualBenchVecEnv` with the same role as `ContinualBenchVecEnv`. Use DLPack to share arrays with
torch SAC, make UTD configurable, and keep the CPU path behind a config flag. *Check: single-task SAC
on `button` (an easy task) reaches a similar success rate to the CPU run, with the same seed budget in
env steps.* Only then run the full 6-task R&D sequence and the sweeps.

### Beginner pitfalls to watch for

- **Recompiling every step.** If a step takes seconds every time, something passed to the jitted
  function is a Python value that changes (like the task name string). Make it static or an array.
- **First-call slowness.** The first `jit` call compiles (I measured 9–13 s on CPU). That's
  normal. Time the **second** call onwards.
- **Silent −1 ids.** `mj_name2id` returns −1 for a missing name and Python happily indexes
  `[-1]`. Assert on every lookup (this codebase already has two of these bugs).
- **float32.** JAX defaults to float32 and the CPU env uses float64. Expect small differences and
  set tolerances on purpose. `jax.config.update("jax_enable_x64", True)` helps when debugging parity
  on CPU, but don't use it for GPU speed runs.
- **`jnp.where` evaluates both sides.** Guard logs, square roots and divisions (see `hamacher_product`).
- **Contact overflow on Warp.** If `naconmax` is too small, contacts get dropped and objects fall
  through each other. Log the actual maximum and leave headroom.

---

## Part 6: Is it worth it?

- **Upside:** thousands of envs per GPU instead of 10 CPU workers, no pickling or pipes, and resets
  that cost nothing. That makes big-batch SAC sweeps (the ManiSkill3-style setup) practical on this
  benchmark.
- **Main risk:** the collision and solver changes needed for speed change the physics, especially for
  grasping (`peg`, `block`). Results on the GPU env won't be directly comparable to the
  original ContinualBench numbers, so report it as a variant ("ContinualBench-MJX") and keep the CPU
  env as the reference.
- **Suggested order of attack:** Phases 1–2 are worth doing even if you stop there, because a
  `jnp`/batched reward and obs module plus parity tests is useful on its own. Phase 3 (collision
  model) tells you early whether the rest is worth it: if contact slots won't come down without breaking the
  demos, move straight to MuJoCo Warp on a GPU before writing the env loop.
