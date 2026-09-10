.. _performance-and-scaling:

Performance and scaling
=======================

Arena can shorten an evaluation in two complementary ways. Within one Run, it can advance many
environments together on one GPU to increase rollout throughput. Across an Experiment, OSMO can
execute independent Runs at the same time on multiple GPUs to reduce the time needed to finish the
complete set of Runs.

An environment-step is one simulation step completed by one environment. A vectorized step
advances every parallel environment in a Run once. For example, one vectorized step with 256
parallel environments completes 256 environment-steps.

Both benchmarks below used the same camera-free workload: the DROID Rubik's-cube-into-bowl task at
the Maple table, the ``zero_action`` policy that sends zero-valued actions, and 300 vectorized
steps per Run.

.. note::

   These are preliminary reference measurements collected for this release. They show how this
   specific workload scaled on the named hardware; they are not performance guarantees for other
   tasks or systems.


.. _performance-parallel-environments:

Parallel environments within one Run
------------------------------------

The single-GPU benchmark ran a fresh Arena process for each environment count on one NVIDIA RTX
5880 Ada Generation GPU with 49,140 MiB of memory. Rollout throughput is the number of parallel
environments divided by the mean time for one vectorized step. It excludes process startup,
environment construction, report generation, and shutdown.

The host also used an Intel Core i9-10920X CPU with 12 cores and 24 threads, 62 GiB of system
memory, Ubuntu 22.04.5, and NVIDIA driver 560.35.05.

.. figure:: ../../images/performance/parallel_environment_throughput.svg
   :alt: Rollout throughput at 1, 64, 256, 512, and 1,024 parallel environments.
   :width: 100%
   :align: center

.. list-table:: Single-GPU rollout results
   :header-rows: 1
   :widths: 25 35 40

   * - Parallel environments
     - Mean vectorized step
     - Rollout throughput
   * - 1
     - 185.0 ms
     - 5.41 environment-steps/s
   * - 64
     - 200.0 ms
     - 319.93 environment-steps/s
   * - 256
     - 234.4 ms
     - 1,092.18 environment-steps/s
   * - 512
     - 280.6 ms
     - 1,824.90 environment-steps/s
   * - 1,024
     - 428.4 ms
     - 2,390.22 environment-steps/s


.. _performance-distributed-runs:

Independent Runs across GPUs with OSMO
--------------------------------------

The distributed benchmark used one Experiment containing eight identical Runs. Each Run created
256 parallel environments, advanced them for 300 steps, and used one NVIDIA L40 GPU. OSMO was
configured to execute at most 1, 2, 4, or 8 Runs at once.

With fewer than eight GPUs, OSMO executed the Runs in consecutive groups. The active Arena
execution time below is the sum of the active time for those groups, from the first Arena process
starting until the last process in each group exited. It includes Arena and Isaac Sim startup,
environment construction, rollout, and shutdown. It excludes OSMO queueing, container-image
downloads, inactive time between groups, and final output collection.

.. figure:: ../../images/performance/distributed_run_speedup.svg
   :alt: Arena execution speedup at 1, 2, 4, and 8 concurrent GPUs.
   :width: 100%
   :align: center

.. list-table:: OSMO distributed-Run results
   :header-rows: 1
   :widths: 20 27 23 15 15

   * - Concurrent Runs and GPUs
     - Scheduling of eight Runs
     - Active Arena execution time
     - Speedup
     - Mean Run duration
   * - 1
     - Eight consecutive Runs
     - 1,255.2 s
     - 1.00x
     - 156.9 s
   * - 2
     - Four groups of two
     - 621.0 s
     - 2.02x
     - 154.6 s
   * - 4
     - Two groups of four
     - 310.2 s
     - 4.05x
     - 154.3 s
   * - 8
     - All eight together
     - 157.8 s
     - 7.95x
     - 154.0 s

Executing all eight Runs at once reduced active Arena execution time from 20 minutes 55 seconds to
2 minutes 38 seconds, a 7.95x speedup. The mean duration of an individual Run changed by less than
2% across the four measurements. All 32 Runs completed successfully; the eight-GPU configuration
used six worker nodes.


Using both scaling axes
-----------------------

The two approaches address different parts of an evaluation and can be combined. Parallel
environments increase the amount of simulation work completed by each Run on its GPU. Distributing
independent Runs lets OSMO execute more of the Experiment at the same time across available GPUs
and worker nodes.


Benchmark scope
---------------

* The single-GPU test ran on a local engineering workstation, not a controlled performance lab
  system.
* The workload did not render cameras or run policy inference. Cameras, policies, scene contents,
  and physics settings can change both throughput and capacity.
* The single-GPU and OSMO benchmarks used different GPU models and software builds. Their absolute
  step times should not be compared directly.
* Arena's component timers use CPU wall-clock time without explicit CUDA synchronization. They are
  rollout diagnostics, not GPU kernel measurements.
* Full OSMO submission time is not used for the distributed speedup because container-image cache
  state differed between submissions.


Tested revisions
----------------

* **Single-GPU benchmark at 1, 64, and 256 environments, August 27, 2026:** Arena `b0cd0b38e
  <https://github.com/isaac-sim/IsaacLab-Arena/commit/b0cd0b38e660637ee5bc7f8c962994cb1cac4852>`_,
  Isaac Lab `af1bab4dc
  <https://github.com/isaac-sim/IsaacLab/commit/af1bab4dc173ba69b08fab779c14ead61d13fd33>`_, and
  the `camera-free benchmark configuration
  <https://github.com/isaac-sim/IsaacLab-Arena/blob/b0cd0b38e660637ee5bc7f8c962994cb1cac4852/isaaclab_arena_environments/experiment_configs/perflab/camera_free_benchmark_experiment.yaml>`_.
* **Single-GPU follow-up at 512 and 1,024 environments, September 9, 2026:** the same Arena and Isaac
  Lab revisions and camera-free workload.
* **OSMO benchmark, September 6, 2026:** Arena `4ee056866
  <https://github.com/isaac-sim/IsaacLab-Arena/commit/4ee056866b0f222fa166561ed46e2b5bace39445>`_,
  Isaac Lab `bb0c8e1b9
  <https://github.com/isaac-sim/IsaacLab/commit/bb0c8e1b9af381bf13064ec3303e17db79e4b6ef>`_, and
  the `OSMO benchmark configuration
  <https://github.com/isaac-sim/IsaacLab-Arena/blob/4ee056866b0f222fa166561ed46e2b5bace39445/isaaclab_arena_environments/experiment_configs/perflab/osmo/camera_free_scaling_validation_experiment.yaml>`_.
