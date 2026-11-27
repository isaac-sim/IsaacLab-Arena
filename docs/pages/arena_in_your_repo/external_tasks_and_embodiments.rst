Your Own Tasks and Embodiments
==============================

The :doc:`previous page <external_environments>` showed how to define an
environment in your repository, external to the Isaac Lab - Arena source tree.
Typical users will also want to define their own tasks and embodiments.
This page describes how that is done, in an external repo. The code comes from
``isaaclab_arena_examples/external_environments/advanced.py``. Copy the file into your package, for
example as ``my_package/isaaclab_arena_environments/my_environment_with_task.py``.

Defining a Custom Task
----------------------

A custom task is defined by subclassing ``TaskBase`` and implementing the required methods.
The excerpt below shows a task that succeeds after a fixed number of steps.
``get_termination_cfg()`` declares its success criteria, failure conditions, and time limit.
This task can be passed to the ``ArenaEnvBuilder`` to create an environment
(see :ref:`putting_it_all_together` below for an example).

.. literalinclude:: ../../../isaaclab_arena_examples/external_environments/advanced.py
   :language: python
   :pyobject: SuccessAfterNStepsTask

Defining a Custom Embodiment
----------------------------

Custom embodiments are defined by subclassing ``EmbodimentBase`` and implementing
the required methods.
In this example, we keep things simple by subclassing an existing embodiment
(``FrankaIKEmbodiment``) and overriding some joint PD gains to produce a more compliant
arm:

.. literalinclude:: ../../../isaaclab_arena_examples/external_environments/advanced.py
   :language: python
   :pyobject: SoftFrankaIKEmbodiment

The ``@register_asset`` decorator registers the class with the
``AssetRegistry`` under its ``registry_name``, ``"franka_ik_soft"``, so it can be fetched
the same way as any built-in embodiment (see :ref:`putting_it_all_together` below for an example).

.. _recording-per-episode-values:

Recording a Custom Value per Episode
------------------------------------

The example also records two values for each episode through ``episode_recorder_terms``, shown below. It
defines ``record_object_final_height``, a function for a value read when the episode ends, and uses Arena's
``ObjectPeakSpeed``, a class derived from ``EpisodeRecorderTerm`` for a value that builds up during the
episode. Each episode's line in the episode results holds the values under the terms' names, for example
``"object_final_height": 0.12, "object_peak_speed": 0.83``.

.. _putting_it_all_together:

Putting It All Together
-----------------------

Here we create an example environment, like in :doc:`external_environments`. This time it passes
the custom task, the custom embodiment, and the two episode recorder terms to ``IsaacLabArenaEnvironment``.

.. literalinclude:: ../../../isaaclab_arena_examples/external_environments/advanced.py
   :language: python
   :start-at: @dataclass

To run it, pass the factory class to ``--external_environment_class_path``, followed by the
environment name and its flags. For example, with a zero-action policy:

.. code-block:: bash

   python isaaclab_arena/evaluation/policy_runner.py \
     --policy_type zero_action \
     --num_episodes 1 \
     --external_environment_class_path my_package.isaaclab_arena_environments.my_environment_with_task.ExternalFrankaTableWithTaskEnvironment \
     franka_table_with_task \
     --object tomato_soup_can

Importing the module also registers the custom embodiment, because ``@register_asset`` runs on
import.

.. note::

    The example is not a built-in environment, so it also runs through
    ``--external_environment_class_path`` from the Arena source tree:

    .. code-block:: bash

        python isaaclab_arena/evaluation/policy_runner.py \
          --viz kit \
          --policy_type zero_action \
          --num_episodes 1 \
          --external_environment_class_path isaaclab_arena_examples.external_environments.advanced.ExternalFrankaTableWithTaskEnvironment \
          franka_table_with_task \
          --object tomato_soup_can
