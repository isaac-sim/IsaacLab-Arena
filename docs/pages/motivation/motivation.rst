Why Isaac Lab-Arena
===================

.. _why-opportunity:

Opportunity
-----------

Simulation makes broad policy evaluation feasible before expensive deployment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Generalist robot policies such as `GR00T <https://developer.nvidia.com/isaac/gr00t>`_ and
`π0 <https://www.physicalintelligence.company/>`_ aim to operate across many tasks, scenes, objects,
embodiments, and deployment conditions. Specialist policies must also remain reliable as deployment
conditions vary.

Evaluating policy robustness requires more than a fixed benchmark suite. Coverage grows combinatorially
across tasks, scenes, embodiments, objects, and environment variations. Lighting, clutter, object
substitutions, and robot morphology can all change policy behavior; limited coverage can favor
policies tuned to benchmark-specific conditions rather than those that generalize.

Simulation makes policy evaluation at this breadth practical, revealing where a policy holds—and where it
breaks—while iteration is still fast and before real-world testing becomes slow and expensive.


.. _why-gap:

Gap
---

The evaluation space scales. Today's evaluation stack does not.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Evaluation coverage grows combinatorially, but most tools still treat every variation as a
standalone environment, every benchmark as a new integration, and every run as a queue. Four
bottlenecks follow.

.. raw:: html

   <section class="arena-gap-story" aria-label="Four limitations of today's evaluation stack">
     <article class="arena-gap-story-row">
       <div class="arena-gap-story-copy">
         <span>01</span>
         <h3>Diversity requires redundant code and effort</h3>
         <p>Each new object or embodiment still means another task configuration—resulting in significant code duplication even when the scene setup, observations, actions, and task logic are largely unchanged.</p>
       </div>
       <div class="arena-gap-proof arena-gap-proof-01" role="img" aria-label="One task configuration grows to four when objects vary and eight when both objects and robot embodiments vary">
         <div class="arena-gap-redundancy">
           <span class="arena-gap-redundancy-label">MANUAL EFFORT + REDUNDANCY</span>
           <svg class="arena-gap-manual-curve" viewBox="0 0 280 130" preserveAspectRatio="none" aria-hidden="true" focusable="false">
             <defs>
               <linearGradient id="arena-manual-effort-area" x1="0" y1="0" x2="0" y2="1">
                 <stop offset="0%" stop-color="#d83b30" stop-opacity=".22" />
                 <stop offset="100%" stop-color="#d83b30" stop-opacity="0" />
               </linearGradient>
               <marker id="arena-manual-effort-arrow" markerWidth="5" markerHeight="5" refX="4.2" refY="2.5" orient="auto">
                 <path d="M0 0 L5 2.5 L0 5 Z" fill="#d32f25" />
               </marker>
             </defs>
             <path class="arena-gap-manual-area" d="M24 114 C108 114 164 108 201 79 C231 56 248 25 263 9 L263 116 L24 116 Z" />
             <path class="arena-gap-manual-line" marker-end="url(#arena-manual-effort-arrow)" d="M24 114 C108 114 164 108 201 79 C231 56 248 25 263 9" />
           </svg>
           <div class="arena-gap-config-cases">
             <section>
               <header><b>Fourier GR-1</b><span>Pick <strong>Banana</strong></span><small>in Kitchen</small></header>
               <div class="arena-gap-env-stack arena-gap-env-stack-1"><b>Isaac Lab<br>environment</b><i>Scene</i><i>Termination</i><i>Events</i><i>Observations</i><i>Actions</i></div>
               <em><strong>1×</strong> configuration</em>
             </section>
             <section>
               <header><b>Fourier GR-1</b><span>Pick <strong>Banana · Apple<br>Broccoli · Carrot</strong></span><small>in Kitchen</small></header>
               <div class="arena-gap-env-stack arena-gap-env-stack-4"><b>Isaac Lab<br>environment</b><i>Scene</i><i>Termination</i><i>Events</i><i>Observations</i><i>Actions</i></div>
               <em><strong>4×</strong> copied</em>
             </section>
             <section>
               <header><b>Fourier GR-1 + Franka</b><span>Pick <strong>Banana · Apple<br>Broccoli · Carrot</strong></span><small>in Kitchen</small></header>
               <div class="arena-gap-env-stack arena-gap-env-stack-8"><b>Isaac Lab<br>environment</b><i>Scene</i><i>Termination</i><i>Events</i><i>Observations</i><i>Actions</i></div>
               <em><strong>8×</strong> copied</em>
             </section>
           </div>
         </div>
       </div>
     </article>
     <article class="arena-gap-story-row">
       <div class="arena-gap-story-copy">
         <span>02</span>
         <h3>Every benchmark rebuilds the eval scaffold</h3>
         <p>Teams recreate policy adapters, inference loops, experiment definitions, recording, result collection, and reports—creating high overhead, fragmented results, and limited comparability.</p>
       </div>
       <div class="arena-gap-proof arena-gap-proof-02" role="img" aria-label="Behavior-1K and RoboDojo each rebuild a custom evaluation scaffold rather than sharing one evaluation framework">
         <div class="arena-gap-scaffolds">
           <section><header><span>BENCHMARK A</span><b>BEHAVIOR-1K</b></header><strong>Custom evaluation scaffold</strong><footer>Isaac Lab / Sim</footer></section>
           <div><b>Duplicated</b><strong>≠</strong><span>Shared</span></div>
           <section><header><span>BENCHMARK B</span><b>RoboDojo</b></header><strong>Custom evaluation scaffold</strong><footer>Isaac Lab / Sim</footer></section>
         </div>
       </div>
     </article>
     <article class="arena-gap-story-row">
       <div class="arena-gap-story-copy">
         <span>03</span>
         <h3>Leaderboards reward overfitting; results are not actionable</h3>
         <p>A frozen task-set score shows whether a policy passed a narrow set of conditions—not whether it is robust or generalizes. It shows what failed, but not where or which environment factor exposed the weakness.</p>
       </div>
       <div class="arena-gap-proof arena-gap-proof-03" role="img" aria-label="A frozen task set samples one point in a much larger operating envelope while GR00T and pi zero leaderboard scores leave why the policies failed and what to fix unknown">
         <div class="arena-gap-frozen-score">
           <section><span>POLICY OPERATING ENVELOPE</span><div class="arena-gap-envelope"><i></i></div><small><b></b> Frozen task set <b></b> Conditions untested</small></section>
           <section><header><span>LEADERBOARD</span><b>SUCCESS</b></header><div><i>GR00T</i><strong>51%</strong></div><div><i>π0</i><strong>50%</strong></div><footer>Frozen task set</footer></section>
           <aside><span><b>WHY IT FAILED</b>Unknown</span><span><b>WHAT TO FIX</b>Unknown</span></aside>
         </div>
       </div>
     </article>
     <article class="arena-gap-story-row">
       <div class="arena-gap-story-copy">
         <span>04</span>
         <h3>Sequential execution forces shallow coverage</h3>
         <p>Sequential runs take too long, so teams compromise on insights, tasks, variations, and seeds to get an answer on schedule.</p>
       </div>
       <div class="arena-gap-proof arena-gap-proof-04" role="img" aria-label="A sequential runner reaches the deadline after evaluating only three of thirty-two conditions">
         <div class="arena-gap-sequential">
           <header><span>SEQUENTIAL RUNS</span><b>DEADLINE</b><em>TIME BUDGET</em></header>
           <div class="arena-gap-runner"><strong>RUNNER</strong><i>01</i><i>02</i><i>03</i><i>04</i><i>05</i><i>06</i><i>07</i></div>
           <section><span>Coverage at deadline</span><div class="arena-gap-coverage"><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i></div><strong>3 of 32<small>conditions evaluated</small></strong></section>
         </div>
       </div>
     </article>
   </section>


.. _why-solution:

Solution
--------

A shared evaluation framework
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Your benchmark defines the tasks and metrics. Isaac Lab-Arena provides the shared system to author
benchmarks, execute policy evaluations, and analyze results, while extending the Isaac Lab
simulation framework and its physics solvers.

Three approaches to scalable, actionable benchmarking
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Compositional approach to environment authoring
"""""""""""""""""""""""""""""""""""""""""""""""

Define scenes, embodiments, and tasks as reusable modules. ``ArenaEnvBuilder`` composes them into a
standard Isaac Lab ``ManagerBasedRLEnvCfg`` at run time, so a new object or embodiment does not
require duplicating the complete task configuration.

.. container:: arena-why-feature-links

   :doc:`Explore environment concepts <../quickstart/arena_env>`


Variational approach to benchmarking
"""""""""""""""""""""""""""""""""""""

Turn a base environment into a controlled sweep across objects, placements, embodiments, and
conditions to analyze policy robustness. Arena records the sampled values with each episode and
computes a joint posterior to reveal which factors impact policy performance.

.. container:: arena-why-feature-links

   :doc:`Explore variation concepts <../quickstart/environment_variations>`

   :doc:`Explore sensitivity analysis concepts <../concepts/concept_sensitivity_analysis>`


Parallel evaluation
"""""""""""""""""""""""""""""""""""""""""""""""""""

Run one policy concurrently across parallel environments, or distribute multi-policy, multi-task
experiments across nodes. Parallel execution makes broad task coverage and deep per-episode
analysis practical within reasonable time.

.. container:: arena-why-feature-links

   :doc:`Explore Arena experiments and parallel environments <../concepts/concept_arena_experiments>`
