# Relation solver benchmarks

`benchmark_relation_solver.py` measures the collision-heavy parts of the relation solver without starting Isaac Sim.
It uses deterministic `DummyObject` scenes and reports median and p95 wall time plus incremental peak CUDA memory.

Run it inside this clone's Arena container:

```bash
/isaac-sim/python.sh benchmarks/benchmark_relation_solver.py
```

The benchmark contains three phases:

- `collision_forward`: AABB extent construction, pair enumeration/stacking, and the collision-loss forward pass.
- `collision_forward_backward`: the collision forward pass plus autograd.
- `solve`: a fixed-iteration `RelationSolver.solve()` call, including all relation losses and optimizer work.

Both dense and sparse initial layouts are included. CUDA scenes with at least 64 collision objects use the grouped BVH
broad phase, while small, CPU, debug, and densely overlapping scenes use dense vectorization. The `selected` column is
the total number of directed pair instances selected across the candidate batch.

## Recording a baseline

Use fixed iterations so convergence does not confound the comparison. This larger sweep exercises the direct-placement
batch size of 10 and the normal placement-pool batch size of 50:

```bash
/isaac-sim/python.sh benchmarks/benchmark_relation_solver.py \
  --object-counts 8 16 32 64 \
  --batch-sizes 1 10 50 \
  --iterations 100 \
  --warmups 5 \
  --repeats 30 \
  --output outputs/benchmarks/relation_solver_main.json
```

Run the same command on an implementation branch and compare it to the recorded baseline:

```bash
/isaac-sim/python.sh benchmarks/benchmark_relation_solver.py \
  --object-counts 8 16 32 64 \
  --batch-sizes 1 10 50 \
  --iterations 100 \
  --warmups 5 \
  --repeats 30 \
  --baseline outputs/benchmarks/relation_solver_main.json \
  --output outputs/benchmarks/relation_solver_candidate.json
```

`speedup` is `baseline median / current median`, so values greater than 1 are faster. The comparison rejects results from
different devices or Torch/CUDA versions. JSON output records the Git commit, branch, dirty-worktree status, device,
Torch/CUDA versions, configuration, raw samples, summary statistics, initial pair count, and initial collision loss.

For CPU measurements, hide CUDA from the process:

```bash
CUDA_VISIBLE_DEVICES="" /isaac-sim/python.sh benchmarks/benchmark_relation_solver.py
```

Performance results are intentionally not pytest assertions. Machine load, GPU model, driver state, and thermal behavior
make fixed timing thresholds unsuitable for the correctness test suite.
