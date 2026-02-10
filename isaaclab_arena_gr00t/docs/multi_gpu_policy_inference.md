# Multi-GPU policy inference (one process per GPU)

**Approach:** One process per GPU via `torch.distributed.run`. Device selection is done **only in Arena** (no changes to the Isaac-GR00T submodule).

---

## Where the model is loaded (no wrapping)

The GR00T policy model is loaded and placed on a single device. There is no DDP or other wrapper.

### 1. Isaac-GR00T (submodule — no changes)

**File:** `submodules/Isaac-GR00T/gr00t/policy/gr00t_policy.py`

```python
# Lines 79-85: model is loaded and moved to the device passed by the caller
model = AutoModel.from_pretrained(model_dir)
model.eval()
model.to(device=device, dtype=torch.bfloat16)
self.model = model
```

The submodule only uses the `device` it is given. All multi-GPU behavior is controlled from Arena.

### 2. Arena: policy construction and device resolution

**File:** `isaaclab_arena_gr00t/policy/gr00t_closedloop_policy.py`

Device is resolved in Arena with **no submodule changes**:

1. **`_resolve_policy_device(config)`** chooses the device in this order:
   - If `LOCAL_RANK` is set (e.g. when launched with `torchrun`) → `cuda:<LOCAL_RANK>`
   - Else if `policy_device_id` is set (CLI or config) → `cuda:<policy_device_id>`
   - Else → `policy_config.policy_device` (e.g. `"cuda"` or `"cuda:0"` from YAML)

2. **`load_policy()`** passes that resolved device to `Gr00tPolicy(..., device=self.device)`.

So when you run with `torchrun --nproc_per_node=N`, each process gets `LOCAL_RANK` set and the policy automatically uses `cuda:0`, `cuda:1`, … without any code changes in the submodule.

---

## One process per GPU with torch.distributed.run

### Step 1: Launch with torchrun

Use one process per GPU. Each process gets its own `LOCAL_RANK`; Arena uses it to set the policy device.

Example (single node, 4 GPUs):

```bash
python -m torch.distributed.run --nproc_per_node=4 your_eval_script.py \
  --policy_config_yaml_path /path/to/config.yaml \
  # other args ...
```

No need to pass `--policy_device` or `--policy_device_id`: the policy will use `cuda:LOCAL_RANK` in each process.

### Step 2: Optional — init process group in your entrypoint

If your script needs the process group for other reasons (e.g. gathering metrics, barrier), init it once at startup:

```python
import os
import torch.distributed as dist

if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
```

Policy device is still driven by `LOCAL_RANK` inside `Gr00tClosedloopPolicy`; you do **not** need to set `config.policy_device` in the entrypoint.

### Step 3: Optional — pin a single GPU (no torchrun)

For a single process on a specific GPU, use device id only:

```bash
python your_eval_script.py --policy_config_yaml_path ... --policy_device_id 1
```

Or pass a full device string:

```bash
python your_eval_script.py --policy_config_yaml_path ... --policy_device cuda:1
```

### Step 4: Independent evals (eval_runner)

The **eval_runner** supports independent evals when launched with `torchrun`:

- **Auto-detection:** If `WORLD_SIZE` > 1, the runner initializes the process group and sets `torch.cuda.set_device(local_rank)`.
- **Job assignment:** Each rank runs only jobs where `job_index % world_size == rank`, so jobs are split across GPUs with no overlap.
- **Policy device:** Each process uses `cuda:LOCAL_RANK` (via `Gr00tClosedloopPolicy._resolve_policy_device()`), so no submodule changes.

Example (4 jobs, 4 GPUs — each rank runs one job):

```bash
python -m torch.distributed.run --nproc_per_node=4 \
  -m isaaclab_arena.evaluation.eval_runner \
  --eval_jobs_config /path/to/jobs.json
```

Recorder outputs use `dataset_{job_name}_rank{rank}` when `world_size` > 1 to avoid path collisions.

---

## Summary

| Item | Detail |
|------|--------|
| **Submodule** | No code changes. Isaac-GR00T only uses the `device` passed from Arena. |
| **Device resolution** | Arena only: `_resolve_policy_device()` in `gr00t_closedloop_policy.py` (LOCAL_RANK → policy_device_id → policy_device). |
| **One process per GPU** | Launch with `torch.distributed.run --nproc_per_node=N`. Each process uses `cuda:LOCAL_RANK` automatically. |
| **Wrapping** | No DDP or DataParallel; one model per process on one device. |
