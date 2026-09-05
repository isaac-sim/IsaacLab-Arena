Welcome to Isaac Lab-Arena!
===========================

.. note::
   This is the development version of Isaac Lab Arena. It contains the newest features but may not
   be fully tested yet. For the tested version, please refer to the `release/0.2.1 branch
   <https://isaac-sim.github.io/IsaacLab-Arena/release/0.2.1/index.html>`_.

.. _overview:

.. raw:: html

   <section class="arena-overview-hero" aria-labelledby="arena-overview-heading">
     <div class="arena-overview-hero__copy">
       <p class="arena-overview-hero__eyebrow">Overview</p>
       <h2 id="arena-overview-heading">NVIDIA Isaac Lab-Arena is an open-source framework for scalable benchmark authoring and robot-policy evaluation in simulation.</h2>
       <p><strong>Not another benchmark or library of benchmarks.</strong> Arena extends <a href="https://isaac-sim.github.io/IsaacLab/main/index.html">Isaac Lab</a> with reusable APIs to author benchmarks, execute evaluations at scale, and analyze results for actionable feedback.</p>
     </div>
     <div class="arena-system-position" role="img" aria-label="Your benchmarks are built on Isaac Lab-Arena, which authors benchmarks, executes evaluations, and analyzes results. Isaac Lab-Arena extends NVIDIA Isaac Lab, which uses PhysX and Newton physics solvers.">
       <div class="arena-system-position__stack">
        <div class="arena-system-position__layer arena-system-position__layer--benchmark">
          <strong>Your benchmarks</strong>
         </div>
         <div class="arena-system-position__connector" aria-hidden="true"><span>built on</span></div>
         <div class="arena-system-position__layer arena-system-position__layer--arena">
           <div>
             <span>Policy evaluation framework</span>
             <strong>NVIDIA Isaac Lab-Arena</strong>
           </div>
           <div class="arena-system-position__capabilities">
             <span><b>Author</b> benchmarks</span>
             <span><b>Execute</b> evaluations</span>
             <span><b>Analyze</b> results</span>
           </div>
         </div>
         <div class="arena-system-position__connector" aria-hidden="true"><span>extends</span></div>
         <div class="arena-system-position__layer arena-system-position__layer--lab">
           <span>Simulation framework</span>
           <div>
             <strong>NVIDIA Isaac Lab</strong>
           </div>
         </div>
         <div class="arena-system-position__connector" aria-hidden="true"><span>solves with</span></div>
         <div class="arena-system-position__layer arena-system-position__layer--physics">
           <span>Physics solvers</span>
           <small class="arena-system-position__foundations">PhysX <i aria-hidden="true">·</i> Newton</small>
         </div>
       </div>
     </div>
   </section>

Features
--------

.. _home-author-benchmarks:

Author benchmarks
^^^^^^^^^^^^^^^^^

.. _home-modular-environment-composition:

.. container:: arena-feature-row arena-feature-rich

   .. grid:: 1 1 2 2
      :gutter: 4
      :class-container: arena-feature-grid

      .. grid-item::
         :columns: 12 12 5 5

         .. container:: arena-feature-copy

            .. rubric:: :doc:`Modular environment composition <pages/concepts/environment/index>`

            Build dynamic, LEGO-like environments from independent scene, object, embodiment, and
            task modules. Swap one without rebuilding the rest.

      .. grid-item::
         :columns: 12 12 7 7

         .. container:: arena-composition-proof

            .. container:: arena-composition-modules

               .. container:: arena-composition-module arena-composition-scene

                  .. rubric:: Scene

                  **Kitchen**

               .. container:: arena-composition-module arena-composition-objects

                  .. rubric:: Objects

                  **Banana + plate**

               .. container:: arena-composition-module arena-composition-embodiment

                  .. rubric:: Embodiment

                  **Franka**

               .. container:: arena-composition-module arena-composition-task

                  .. rubric:: Task

                  **Pick → place**

            .. container:: arena-composition-arrow

               ``ArenaEnvBuilder``

            .. container:: arena-composition-family

               .. rubric:: Environment family

               .. image:: images/landing/composable-environment-family-eight-clean.webp
                  :width: 100%
                  :alt: An Arena environment family created by recombining objects and embodiments from the same environment modules
                  :loading: lazy

.. _home-relational-object-placement:

.. container:: arena-feature-row arena-feature-rich

   .. grid:: 1 1 2 2
      :gutter: 4
      :class-container: arena-feature-grid

      .. grid-item::
         :columns: 12 12 5 5

         .. container:: arena-feature-copy

            .. rubric:: :doc:`Relational object placement <pages/concepts/concept_object_and_robot_placement>`

            Define layouts with semantic spatial relations rather than hand-coded poses. Arena
            solves and validates candidate placements against object geometry, collisions, and task
            constraints.

      .. grid-item::
         :columns: 12 12 7 7

         .. container:: arena-placement-proof arena-placement-giggles

            .. container:: arena-placement-stage arena-placement-coordinate

               .. container:: arena-placement-stage-header

                  .. rubric:: Before Arena

                  **Hard-coded coordinates**

               .. container:: arena-placement-stage-body

                  ``microwave``

                  ``x 0.82 · y −0.14 · z 0.91``

               .. container:: arena-placement-stage-footer

                  Recalculate poses when the scene changes.

            .. container:: arena-panel-connector

               →

            .. container:: arena-placement-stage arena-placement-relations

               .. container:: arena-placement-stage-header

                  .. rubric:: With Arena

                  **Describe spatial intent**

               .. container:: arena-placement-stage-body

                  ``plate`` ``On`` ``table``

                  ``banana`` ``On`` ``plate``

               .. container:: arena-placement-stage-footer

                  Relations remain readable and reusable.

            .. container:: arena-panel-connector

               →

            .. container:: arena-placement-resolution

               .. container:: arena-placement-resolution-header

                  .. rubric:: Arena resolves + builds

                  **Intent becomes a valid simulation environment**

               .. container:: arena-placement-resolution-body

                  .. container:: arena-placement-figure arena-placement-solver

                     .. video:: images/landing/relational-placement-solver.mp4
                        :loop:
                        :muted:
                        :playsinline:
                        :nocontrols:
                        :preload: none
                        :poster: _images/relational-placement-solver.webp

                     **Placement solver**

                  .. container:: arena-placement-inside-connector

                     →

                  .. container:: arena-placement-figure arena-placement-environment

                     .. video:: images/landing/relational-placement-resolved.mp4
                        :loop:
                        :muted:
                        :playsinline:
                        :nocontrols:
                        :preload: none
                        :poster: _images/relational-placement-resolved.webp

                     **Environment ready for evaluation**

.. _home-agentic-environment-generation:

.. container:: arena-feature-row arena-feature-rich

   .. grid:: 1 1 2 2
      :gutter: 4
      :class-container: arena-feature-grid

      .. grid-item::
         :columns: 12 12 5 5

         .. container:: arena-feature-copy

            .. container:: arena-feature-heading

               .. rubric:: :doc:`Agentic environment generation <pages/concepts/agentic_environment_generation/index>`

               .. raw:: html

                  <span class="arena-experimental-chip" tabindex="0" aria-describedby="arena-agentic-experimental-note">
                    <i aria-hidden="true"></i>Experimental<b aria-hidden="true">?</b>
                    <span id="arena-agentic-experimental-note" role="tooltip">The underlying agents and architecture may change in future versions.</span>
                  </span>

            Describe the benchmark you want in natural language. The agent infers constraints,
            produces an editable spec, fetches existing SimReady USDs, and builds the Arena
            environment(s).

      .. grid-item::
         :columns: 12 12 7 7

         .. container:: arena-agentic-proof arena-agentic-giggles

            .. raw:: html

               <div class="arena-agentic-switcher" role="tablist" aria-label="Select an agentic environment generation example">
                 <span>Select an example</span>
                 <button type="button" class="arena-agentic-tab arena-agentic-tab-active" role="tab" aria-selected="true" data-arena-agentic-tab="domestic"><b>01</b> Domestic</button>
                 <button type="button" class="arena-agentic-tab" role="tab" aria-selected="false" data-arena-agentic-tab="industrial"><b>02</b> Industrial</button>
               </div>

            .. container:: arena-agentic-example arena-agentic-example-domestic arena-agentic-example-active

               .. container:: arena-agentic-demo

                  .. container:: arena-agentic-prompt

                     .. rubric:: Input prompt

                     droid pick up the banana and put it on the plate. Using maple table
                     background. Other objects on the table as distractors: two bagels, bowl,
                     with positions randomized

                  .. container:: arena-agentic-connector

                     →

                  .. container:: arena-agentic-output

                     .. rubric:: Output: Ready-to-evaluate Arena environments

                     .. image:: images/landing/tabletop-agentic-env-banana-bagel-plate.webp
                        :width: 100%
                        :alt: Arena environments generated for a DROID banana-to-plate task with distractors
                        :loading: lazy

            .. container:: arena-agentic-example arena-agentic-example-industrial

               .. container:: arena-agentic-demo

                  .. container:: arena-agentic-prompt

                     .. rubric:: Input prompt

                     Two bins on maple table. ``container_f24`` is on the left of ``bin_b04``.
                     droid put the spring clamp in the right bin. Other objects on the table as
                     distractors: two hammers, cordless drill.

                  .. container:: arena-agentic-connector

                     →

                  .. container:: arena-agentic-output

                     .. rubric:: Output: Ready-to-evaluate Arena environments

                     .. image:: images/landing/agentic-industrial-spring-clamp.webp
                        :width: 100%
                        :alt: Arena industrial environment generated for a spring-clamp placement task
                        :loading: lazy

.. _home-variation-system:

.. container:: arena-feature-row arena-feature-rich

   .. grid:: 1 1 2 2
      :gutter: 4
      :class-container: arena-feature-grid

      .. grid-item::
         :columns: 12 12 5 5

         .. container:: arena-feature-copy

            .. rubric:: :doc:`Variation system <pages/concepts/variations/index>`

            Turn one environment into a controlled sweep of conditions. Define ranges and
            distributions once; sample them at build time or reset.

      .. grid-item::
         :columns: 12 12 7 7

         .. container:: arena-variation-proof arena-variation-giggles

            .. container:: arena-variation-media

               .. container:: arena-motion-tile

                  .. video:: images/landing/hdr-web.mp4
                     :loop:
                     :muted:
                     :playsinline:
                     :nocontrols:
                     :preload: none
                     :poster: _images/hdr-poster.webp

                  **HDR background**

               .. container:: arena-motion-tile

                  .. video:: images/landing/color-web.mp4
                     :loop:
                     :muted:
                     :playsinline:
                     :nocontrols:
                     :preload: none
                     :poster: _images/color-poster.webp

                  **Light color**

               .. container:: arena-motion-tile

                  .. video:: images/landing/temperature-web.mp4
                     :loop:
                     :muted:
                     :playsinline:
                     :nocontrols:
                     :preload: none
                     :poster: _images/temperature-poster.webp

                  **Color temperature**

               .. container:: arena-motion-tile

                  .. video:: images/landing/shadows-web.mp4
                     :loop:
                     :muted:
                     :playsinline:
                     :nocontrols:
                     :preload: none
                     :poster: _images/shadows-poster.webp

                  **Light direction**

.. _home-isaac-lab-interoperability:

.. container:: arena-feature-row arena-feature-rich

   .. grid:: 1 1 2 2
      :gutter: 4
      :class-container: arena-feature-grid

      .. grid-item::
         :columns: 12 12 5 5

         .. container:: arena-feature-copy

            .. rubric:: Isaac Lab interoperability

            Author benchmarks and execute policy evaluations in Arena. Register the Arena-authored environment in Isaac Lab
            with ease to perform data collection or policy learning in Isaac Lab.

      .. grid-item::
         :columns: 12 12 7 7

         .. container:: arena-interop-proof arena-interop-giggles

            .. raw:: html

               <div class="arena-interop-flow" aria-label="An Isaac Lab-Arena benchmark can be evaluated immediately, or plugged into Isaac Lab with an Environment registration callback, learned with Isaac Teleop, Mimic, or reinforcement learning, and then evaluated in Isaac Lab-Arena.">
                 <article class="arena-interop-endpoint arena-interop-author">
                   <img src="_images/composable-environment-family-eight-clean.webp" alt="Arena benchmark environment family" loading="lazy">
                   <div><small>Isaac Lab-Arena</small><strong>Author benchmark</strong></div>
                 </article>
                 <div class="arena-interop-branch arena-interop-branch-out" aria-hidden="true">
                   <svg viewBox="0 0 28 226" preserveAspectRatio="none">
                     <path d="M0 113 H9 V31 H28"></path>
                     <path d="M9 113 V150 H28"></path>
                     <path class="arrowhead" d="M24 27 L28 31 L24 35"></path>
                     <path class="arrowhead" d="M24 146 L28 150 L24 154"></path>
                   </svg>
                 </div>
                 <div class="arena-interop-paths">
                   <section class="arena-interop-path arena-interop-direct">
                     <header><small>Path 1</small><strong>Evaluate now</strong></header>
                     <i class="arena-interop-route-track" aria-hidden="true"><span></span></i>
                   </section>
                   <section class="arena-interop-path arena-interop-learn">
                     <header><small>Path 2</small><strong>Learn, then evaluate</strong></header>
                     <div>
                       <article><small>Environment registration callback</small><strong>Plug environment into Isaac Lab</strong></article>
                       <i class="arena-interop-route-track" aria-hidden="true"><span></span></i>
                       <article><small>Isaac Lab</small><strong>Learn in Isaac Lab</strong><span>Isaac Teleop · Mimic · Reinforcement Learning</span></article>
                     </div>
                   </section>
                 </div>
                 <div class="arena-interop-branch arena-interop-branch-in" aria-hidden="true">
                   <svg viewBox="0 0 28 226" preserveAspectRatio="none">
                     <path d="M0 31 H19 V113 H28"></path>
                     <path d="M0 150 H19 V113"></path>
                     <path class="arrowhead" d="M24 109 L28 113 L24 117"></path>
                   </svg>
                 </div>
                 <article class="arena-interop-endpoint arena-interop-evaluate">
                   <img src="_images/composable-environment-family-eight-clean.webp" alt="The same Arena environment family ready for policy evaluation" loading="lazy">
                   <div><small>Isaac Lab-Arena</small><strong>Evaluate policy</strong></div>
                 </article>
               </div>

.. _home-execute-evaluations:

Execute large-scale parallel policy evaluations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. _home-large-scale-parallel-environments:

.. container:: arena-feature-row arena-feature-rich

   .. grid:: 1 1 2 2
      :gutter: 4
      :class-container: arena-feature-grid

      .. grid-item::
         :columns: 12 12 5 5

         .. container:: arena-feature-copy

            .. rubric:: :doc:`Large-scale parallel environments <pages/concepts/concept_arena_experiments>`

            Evaluate one policy concurrently across thousands of heterogeneous (object-level)
            environments instead of sequential rollouts to speed up policy evaluation.
            `Lightwheel case study <https://lightwheel.ai/media/il-arena-benchmark-study>`_
            reports 10x faster despite using higher fidelity assets.

      .. grid-item::
         :columns: 12 12 7 7

         .. container:: arena-parallel-proof

            .. image:: images/landing/groot-parallel-environments.webp
               :width: 100%
               :alt: Many GR00T robot workcells evaluated in parallel

.. _home-large-scale-multi-node-evaluation:

.. container:: arena-feature-row arena-feature-rich

   .. grid:: 1 1 2 2
      :gutter: 4
      :class-container: arena-feature-grid

      .. grid-item::
         :columns: 12 12 5 5

         .. container:: arena-feature-copy

            .. rubric:: :doc:`Large-scale multi-node evaluation <pages/example_workflows/multi_node_evaluation/multi_node_evaluation>`

            Define experiments with multiple policies and tasks. Run them locally or use an
            orchestrator such as OSMO to distribute across multi-node compute. Arena returns
            aggregate metrics for a high-level summary and per-episode results for detailed
            analysis.

      .. grid-item::
         :columns: 12 12 7 7

         .. container:: arena-multinode-proof

            .. container:: arena-experiment-flow

               .. container:: arena-experiment-stage arena-experiment-define

                  .. rubric:: Define experiment

                  .. raw:: html

                     <dl class="arena-experiment-definition-list">
                       <div><dt>Tasks</dt><dd>20 RoboLab tasks</dd></div>
                       <div><dt>Policies</dt><dd>π0.5 · Cosmos</dd></div>
                       <div><dt>Experiment</dt><dd>100 episodes / task</dd></div>
                     </dl>

               .. container:: arena-experiment-arrow

                  →

               .. container:: arena-experiment-stage arena-experiment-run

                  .. rubric:: Run evaluations

                  **Local or distributed compute**

                  .. container:: arena-experiment-envs arena-experiment-running-envs

                     .. container:: arena-experiment-running-env

                        .. video:: images/landing/main-big-pumpkin.mp4
                           :loop:
                           :muted:
                           :playsinline:
                           :nocontrols:
                           :preload: none
                           :poster: _images/main-big-pumpkin.webp

                        **Big pumpkin in bin**

                     .. container:: arena-experiment-running-env

                        .. video:: images/landing/main-mouse-keyboard.mp4
                           :loop:
                           :muted:
                           :playsinline:
                           :nocontrols:
                           :preload: none
                           :poster: _images/main-mouse-keyboard.webp

                        **Mouse on keyboard**

                     .. container:: arena-experiment-running-env

                        .. video:: images/landing/main-small-pumpkin.mp4
                           :loop:
                           :muted:
                           :playsinline:
                           :nocontrols:
                           :preload: none
                           :poster: _images/main-small-pumpkin.webp

                        **Small pumpkin in bin**

                     .. container:: arena-experiment-running-env

                        .. video:: images/landing/main-mustard-left-bin.mp4
                           :loop:
                           :muted:
                           :playsinline:
                           :nocontrols:
                           :preload: none
                           :poster: _images/main-mustard-left-bin.webp

                        **Mustard in left bin**

                  Experiment Runner · orchestrator such as OSMO

               .. container:: arena-experiment-arrow

                  →

               .. container:: arena-experiment-stage arena-experiment-results

                  .. rubric:: Collate results

                  **One combined result**

                  .. raw:: html

                     <div class="arena-collated-chart" role="img" aria-label="Task success comparison for pi zero point five and Cosmos across ten representative RoboLab tasks">
                       <header>
                         <span>Task success · 10 of 20 tasks shown</span>
                         <small><i class="pi"></i>π0.5 <i class="cosmos"></i>Cosmos</small>
                       </header>
                       <div class="arena-collated-plot">
                         <div class="arena-collated-scale" aria-hidden="true"><span>100</span><span>50</span><span>0</span></div>
                         <div class="arena-collated-grid" aria-hidden="true"><i></i><i></i><i></i></div>
                         <div class="arena-collated-bars">
                           <article title="Banana in bowl: π0.5 94%, Cosmos 98%"><div><i class="pi" style="height:94%"></i><i class="cosmos" style="height:98%"></i></div><small>banana<br>in bowl</small></article>
                           <article title="Banana on plate: π0.5 100%, Cosmos 90%"><div><i class="pi" style="height:100%"></i><i class="cosmos" style="height:90%"></i></div><small>banana<br>on plate</small></article>
                           <article title="Big pumpkin in bin: π0.5 58%, Cosmos 2%"><div><i class="pi" style="height:58%"></i><i class="cosmos" style="height:2%"></i></div><small>big pumpkin<br>in bin</small></article>
                           <article title="Bowl in bin: π0.5 99%, Cosmos 79%"><div><i class="pi" style="height:99%"></i><i class="cosmos" style="height:79%"></i></div><small>bowl<br>in bin</small></article>
                           <article title="Butter above raisin: π0.5 9%, Cosmos 5%"><div><i class="pi" style="height:9%"></i><i class="cosmos" style="height:5%"></i></div><small>butter above<br>raisin</small></article>
                           <article title="Canned food in bin: π0.5 21%, Cosmos 7%"><div><i class="pi" style="height:21%"></i><i class="cosmos" style="height:7%"></i></div><small>canned food<br>in bin</small></article>
                           <article title="Clamp in right bin: π0.5 19%, Cosmos 1%"><div><i class="pi" style="height:19%"></i><i class="cosmos" style="height:1%"></i></div><small>clamp in<br>right bin</small></article>
                           <article title="Coffee pot in bin: π0.5 14%, Cosmos 22%"><div><i class="pi" style="height:14%"></i><i class="cosmos" style="height:22%"></i></div><small>coffee pot<br>in bin</small></article>
                           <article title="Packing boxes: π0.5 13%, Cosmos 0%"><div><i class="pi" style="height:13%"></i><i class="cosmos" style="height:0%"></i></div><small>packing<br>boxes</small></article>
                           <article title="Packing cans: π0.5 49%, Cosmos 8%"><div><i class="pi" style="height:49%"></i><i class="cosmos" style="height:8%"></i></div><small>packing<br>cans</small></article>
                         </div>
                       </div>
                     </div>

            .. container:: arena-video-poster-sources

               .. image:: images/landing/relational-placement-solver.webp
                  :alt: Placement solver animation poster
                  :loading: lazy

               .. image:: images/landing/relational-placement-resolved.webp
                  :alt: Resolved placement animation poster
                  :loading: lazy

               .. image:: images/landing/hdr-poster.webp
                  :alt: HDR background variation poster
                  :loading: lazy

               .. image:: images/landing/color-poster.webp
                  :alt: Light color variation poster
                  :loading: lazy

               .. image:: images/landing/temperature-poster.webp
                  :alt: Color temperature variation poster
                  :loading: lazy

               .. image:: images/landing/shadows-poster.webp
                  :alt: Light direction variation poster
                  :loading: lazy

               .. image:: images/landing/main-big-pumpkin.webp
                  :alt: Big pumpkin in bin evaluation environment poster
                  :loading: lazy

               .. image:: images/landing/main-mouse-keyboard.webp
                  :alt: Mouse on keyboard evaluation environment poster
                  :loading: lazy

               .. image:: images/landing/main-small-pumpkin.webp
                  :alt: Small pumpkin in bin evaluation environment poster
                  :loading: lazy

               .. image:: images/landing/main-mustard-left-bin.webp
                  :alt: Mustard in left bin evaluation environment poster
                  :loading: lazy

               .. image:: images/landing/predicate-progress-rollouts.webp
                  :alt: Predicate progress rollout animation poster
                  :loading: lazy


.. _home-policy-client-server:

.. container:: arena-feature-row arena-feature-rich

   .. grid:: 1 1 2 2
      :gutter: 4
      :class-container: arena-feature-grid

      .. grid-item::
         :columns: 12 12 5 5

         .. container:: arena-feature-copy

            .. rubric:: :doc:`Policy client–server <pages/concepts/policy/index>`

            Run leading foundation models GR00T, π0.5, or bring your own policy behind a server.
            Arena exchanges observations and actions through one client contract across processes,
            GPUs, or machines.

      .. grid-item::
         :columns: 12 12 7 7

         .. container:: arena-policy-proof

            .. raw:: html

               <div class="arena-policy-runtime-flow" aria-label="A separate policy runtime exchanges observations and actions with Isaac Lab-Arena through a shared client contract.">
                 <section class="arena-policy-runtime arena-policy-external">
                   <header><strong>Your runtime</strong><small>Separate process</small></header>
                   <div class="arena-policy-models">
                     <article><i aria-hidden="true"></i><span><small>Policy server</small><strong>GR00T</strong></span></article>
                     <article><i aria-hidden="true"></i><span><small>Policy server</small><strong>π0.5</strong></span></article>
                     <article><i aria-hidden="true"></i><span><small>Policy server</small><strong>Your policy</strong></span></article>
                   </div>
                 </section>
                 <div class="arena-policy-contract">
                   <header><strong>Shared client contract</strong><small>Network boundary</small></header>
                   <div class="arena-policy-lane arena-policy-observations"><i><span></span></i><strong>Observations</strong></div>
                   <div class="arena-policy-lane arena-policy-actions"><strong>Actions</strong><i><span></span></i></div>
                 </div>
                 <section class="arena-policy-runtime arena-policy-arena">
                   <header><strong>Isaac Lab-Arena</strong><small>Evaluation runtime</small></header>
                   <div class="arena-policy-workers" aria-hidden="true"><i></i><i></i><i></i><span></span></div>
                   <div class="arena-policy-capabilities"><span>Tasks</span><span>Scenes</span><span>Metrics</span></div>
                   <footer>No policy dependencies in benchmark</footer>
                 </section>
               </div>

.. _home-analyze-results:

Analyze robustness and why policies fail
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. _home-subtask-predicates:

.. container:: arena-feature-row arena-feature-rich

   .. grid:: 1 1 2 2
      :gutter: 4
      :class-container: arena-feature-grid

      .. grid-item::
         :columns: 12 12 5 5

         .. container:: arena-feature-copy

            .. rubric:: :doc:`Subtask predicates · The where <pages/concepts/task/concept_progress_tracking_design>`

            Track fine-grained milestones within each subtask—for example, decomposing
            pick-and-place into grasp, lift, transport, and place—to see where a policy fails.

      .. grid-item::
         :columns: 12 12 7 7

         .. container:: arena-predicate-proof

            .. container:: arena-predicate-before

               **Before Arena** — End-to-end result only. Success: **20%**. Failure stage:
               **unknown**.

            .. container:: arena-predicate-after

               .. container:: arena-predicate-after-header

                  **After Arena · predicates** — Track progress within every rollout and see where
                  success drops.

               .. container:: arena-predicate-evidence

                  .. container:: arena-predicate-rollout

                     .. video:: images/landing/predicate-progress-rollouts.mp4
                        :loop:
                        :muted:
                        :playsinline:
                        :nocontrols:
                        :preload: none
                        :poster: _images/predicate-progress-rollouts.webp

                  .. container:: arena-predicate-readout

                     .. container:: arena-predicate-stages

                        .. container:: arena-predicate-stage

                           **Grasp**

                           **30.3%**

                        .. container:: arena-predicate-stage arena-predicate-stage-failure

                           **Lift**

                           **22.9%**

                        .. container:: arena-predicate-stage

                           **Transport**

                           **21.3%**

                        .. container:: arena-predicate-stage

                           **Place**

                           **20.0%**

                     **Failure stage: Lift.** 75 rollouts grasp the object but never lift it.

.. _home-sensitivity-analysis:

.. container:: arena-feature-row arena-feature-rich

   .. grid:: 1 1 2 2
      :gutter: 4
      :class-container: arena-feature-grid

      .. grid-item::
         :columns: 12 12 5 5

         .. container:: arena-feature-copy

            .. rubric:: :doc:`Sensitivity analysis · The why <pages/concepts/concept_sensitivity_analysis>`

            Perturb environment factors to reveal policy robustness and get actionable feedback
            that can inform the next round of targeted policy learning.

      .. grid-item::
         :columns: 12 12 7 7

         .. container:: arena-sensitivity-proof

            .. container:: arena-sensitivity-before

               **Before Arena** — End-to-end result only. Success: **20%**. Root cause:
               **unknown**.

            .. container:: arena-sensitivity-after

               .. container:: arena-sensitivity-after-header

                  **After Arena · sensitivity analysis** — View which factors the policy is
                  sensitive to. The same failure can have different root causes and require
                  different fixes.

               .. container:: arena-sensitivity-evidence

                  .. raw:: html

                     <div class="arena-sensitivity-native" role="img" aria-label="Arena sensitivity analysis compares two policies across controlled factors. Policy 1 is sensitive to wrist-camera displacement, while Policy 2 is sensitive to low-light conditions.">

                       <div class="arena-sensitivity-policy-row">
                         <div class="arena-sensitivity-policy-id">
                           <span>Policy</span>
                           <strong>Policy 1</strong>
                           <em>Lift fails</em>
                         </div>

                         <div class="arena-sensitivity-factor is-stable">
                           <div class="arena-sensitivity-factor-header"><strong>Cube mass</strong><span>Stable</span></div>
                           <svg viewBox="0 0 180 72" role="presentation" aria-hidden="true">
                             <path class="arena-chart-grid" d="M8 16H172M8 36H172M8 56H172" />
                             <path class="arena-chart-axis" d="M8 8V60H172" />
                             <path class="arena-chart-robust" d="M8 38 C42 37, 62 39, 92 37 S142 38, 172 37" />
                           </svg>
                           <div class="arena-sensitivity-axis"><span>0.1 kg</span><span>0.5 kg</span><span>0.9 kg</span></div>
                         </div>

                         <div class="arena-sensitivity-factor is-stable">
                           <div class="arena-sensitivity-factor-header"><strong>Lighting</strong><span>Stable</span></div>
                           <svg viewBox="0 0 180 72" role="presentation" aria-hidden="true">
                             <path class="arena-chart-grid" d="M8 16H172M8 36H172M8 56H172" />
                             <path class="arena-chart-axis" d="M8 8V60H172" />
                             <path class="arena-chart-robust" d="M8 39 C38 38, 68 39, 96 38 S144 39, 172 38" />
                           </svg>
                           <div class="arena-sensitivity-axis"><span>50%</span><span>100%</span><span>150%</span></div>
                         </div>

                         <div class="arena-sensitivity-factor is-sensitive">
                           <div class="arena-sensitivity-factor-header"><strong>Wrist cam Y</strong><span>Sensitive</span></div>
                           <svg viewBox="0 0 180 72" role="presentation" aria-hidden="true">
                             <path class="arena-chart-grid" d="M8 16H172M8 36H172M8 56H172" />
                             <path class="arena-chart-axis" d="M8 8V60H172" />
                             <path class="arena-chart-robust" d="M8 30 C42 29, 74 30, 106 30" />
                             <path class="arena-chart-sensitive" d="M106 30 C126 31, 129 34, 137 44 S150 59, 172 62" />
                             <path class="arena-chart-threshold" d="M123 8V61" />
                             <circle class="arena-chart-point" cx="123" cy="32" r="4" />
                           </svg>
                           <div class="arena-sensitivity-axis"><span>−5 cm</span><b>Fail &gt; +2 cm</b><span>+5 cm</span></div>
                         </div>

                         <div class="arena-sensitivity-fix">
                           <span>Analysis · Policy 1</span>
                           <strong>Sensitive to wrist-camera Y displacement</strong>
                           <p>The failure cliff identifies a robustness gap and enables targeted policy improvement.</p>
                         </div>
                       </div>

                       <div class="arena-sensitivity-policy-row">
                         <div class="arena-sensitivity-policy-id">
                           <span>Policy</span>
                           <strong>Policy 2</strong>
                           <em>Lift fails</em>
                         </div>

                         <div class="arena-sensitivity-factor is-stable">
                           <div class="arena-sensitivity-factor-header"><strong>Wrist cam Y</strong><span>Stable</span></div>
                           <svg viewBox="0 0 180 72" role="presentation" aria-hidden="true">
                             <path class="arena-chart-grid" d="M8 16H172M8 36H172M8 56H172" />
                             <path class="arena-chart-axis" d="M8 8V60H172" />
                             <path class="arena-chart-robust" d="M8 37 C38 36, 69 38, 98 37 S145 38, 172 37" />
                           </svg>
                           <div class="arena-sensitivity-axis"><span>−5 cm</span><span>0</span><span>+5 cm</span></div>
                         </div>

                         <div class="arena-sensitivity-factor is-stable">
                           <div class="arena-sensitivity-factor-header"><strong>Cube mass</strong><span>Stable</span></div>
                           <svg viewBox="0 0 180 72" role="presentation" aria-hidden="true">
                             <path class="arena-chart-grid" d="M8 16H172M8 36H172M8 56H172" />
                             <path class="arena-chart-axis" d="M8 8V60H172" />
                             <path class="arena-chart-robust" d="M8 38 C42 37, 64 38, 94 37 S143 38, 172 37" />
                           </svg>
                           <div class="arena-sensitivity-axis"><span>0.1 kg</span><span>0.5 kg</span><span>0.9 kg</span></div>
                         </div>

                         <div class="arena-sensitivity-factor is-sensitive">
                           <div class="arena-sensitivity-factor-header"><strong>Lighting</strong><span>Sensitive</span></div>
                           <svg viewBox="0 0 180 72" role="presentation" aria-hidden="true">
                             <path class="arena-chart-grid" d="M8 16H172M8 36H172M8 56H172" />
                             <path class="arena-chart-axis" d="M8 8V60H172" />
                             <path class="arena-chart-sensitive" d="M8 61 C28 59, 39 52, 51 41 S67 29, 83 29" />
                             <path class="arena-chart-robust" d="M83 29 C111 28, 141 29, 172 28" />
                             <path class="arena-chart-threshold" d="M61 8V61" />
                             <circle class="arena-chart-point" cx="61" cy="34" r="4" />
                           </svg>
                           <div class="arena-sensitivity-axis"><span>50%</span><b>Fail &lt; 70%</b><span>150%</span></div>
                         </div>

                         <div class="arena-sensitivity-fix">
                           <span>Analysis · Policy 2</span>
                           <strong>Sensitive to low-light conditions</strong>
                           <p>The failure cliff identifies a robustness gap and enables targeted policy improvement.</p>
                         </div>
                       </div>

                       <div class="arena-sensitivity-native-footer">
                         <span><i class="is-green"></i>Flat response = robust</span>
                         <span><i class="is-red"></i>Red cliff = sensitive</span>
                       </div>
                     </div>

Why Isaac Lab-Arena
-------------------

.. container:: arena-why

   .. container:: arena-why-story

      .. raw:: html

         <div class="arena-why-copy">
           <p><strong>Simulation can provide early feedback before expensive real-world deployment.</strong></p>
           <p>But as policy evaluation expands across tasks, scenes, conditions, and robot embodiments,<span class="arena-why-pause">standalone environments, custom scripts, and sequential runs do not scale.</span></p>
         </div>

      .. container:: arena-why-action

         :doc:`Read the full story <pages/motivation/motivation>`


Support
-------

.. grid:: 1 2 4 4
   :gutter: 2
   :class-container: arena-resource-grid

   .. grid-item-card:: Questions & Ideas
      :link: https://github.com/isaac-sim/IsaacLab-Arena/discussions
      :link-type: url
      :shadow: none

      :octicon:`comment-discussion;1.35em` GitHub Discussions

   .. grid-item-card:: Bug Reports
      :link: https://github.com/isaac-sim/IsaacLab-Arena/issues
      :link-type: url
      :shadow: none

      :octicon:`issue-opened;1.35em` GitHub Issues

   .. grid-item-card:: Community Chat
      :link: https://discord.com/invite/nvidiaomniverse
      :link-type: url
      :shadow: none

      :octicon:`broadcast;1.35em` Omniverse Discord

   .. grid-item-card:: Isaac Sim Questions
      :link: https://forums.developer.nvidia.com/c/agx-autonomous-machines/isaac/67
      :link-type: url
      :shadow: none

      :octicon:`people;1.35em` NVIDIA Forums


Project resources
-----------------

.. grid:: 1 2 4 4
   :gutter: 2
   :class-container: arena-resource-grid

   .. grid-item-card:: Contributing
      :link: https://github.com/isaac-sim/IsaacLab-Arena#contributing
      :link-type: url
      :shadow: none

      :octicon:`repo-push;1.35em` Contribution workflow

   .. grid-item-card:: Publishing Your Own Benchmark
      :link: https://github.com/isaac-sim/IsaacLab-Arena#publishing-your-own-benchmark
      :link-type: url
      :shadow: none

      :octicon:`repo;1.35em` Publishing workflow

   .. grid-item-card:: Citation Instructions
      :link: https://github.com/isaac-sim/IsaacLab-Arena#citation
      :link-type: url
      :shadow: none

      :octicon:`book;1.35em` Citation guidance

   .. grid-item-card:: License
      :link: https://github.com/isaac-sim/IsaacLab-Arena#license
      :link-type: url
      :shadow: none

      :octicon:`law;1.35em` Apache 2.0 license


Acknowledgements
----------------

Isaac Lab-Arena builds on `NVIDIA Isaac Lab <https://github.com/isaac-sim/IsaacLab>`_, with
the evaluation and task layers designed in close collaboration with Lightwheel. We thank the
Isaac Lab team and the broader robotics community for their foundational work.

Isaac Lab-Arena was built in collaboration with the authors of Robolab (`website
<https://research.nvidia.com/labs/srl/projects/robolab/>`_, `paper
<https://arxiv.org/abs/2604.09860>`_).


.. Documentation navigation is retained for the sidebar but hidden from the landing page.

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Isaac Lab-Arena

   Overview <self>
   Why Isaac Lab-Arena <pages/motivation/motivation>

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Set Up

   pages/quickstart/installation

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Getting Started

   pages/quickstart/arena_env
   pages/quickstart/arena_experiment
   pages/quickstart/environment_variations
   pages/quickstart/running_a_real_policy/index

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Arena in Your Repo

   pages/arena_in_your_repo/index

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Example Workflows

   pages/example_workflows/example_environments
   pages/example_workflows/analysis/index
   pages/example_workflows/agentic_env_gen/index
   pages/example_workflows/imitation_learning/index
   pages/example_workflows/reinforcement_learning_workflows/index

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Concepts

   pages/concepts/environment/index
   pages/concepts/agentic_environment_generation/index
   pages/concepts/scene/index
   pages/concepts/task/index
   pages/concepts/embodiment/index
   pages/concepts/concept_object_and_robot_placement
   pages/concepts/policy/index
   pages/concepts/concept_arena_experiments
   pages/concepts/variations/index
   pages/concepts/concept_sensitivity_analysis

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Advanced

   pages/advanced/private_omniverse
   pages/advanced/assets_management
   pages/quickstart/jupyter_notebooks

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: References

   pages/references/release_notes
   pages/references/citing_us

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Development Team Internal

   pages/development_team_internal/index
