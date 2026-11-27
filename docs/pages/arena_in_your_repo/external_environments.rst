Defining Environments in Your Repository
========================================

You can define an environment in your own package and run it with Arena's runners, without
changing Arena. Like a built-in environment, it has two classes:

* A config dataclass that subclasses ``ArenaEnvironmentCfg``. Its fields are the values a user can
  set. Each environment needs its own config class.
* A factory class that subclasses ``ArenaEnvironmentFactory[YourCfg]``, sets a unique ``name``,
  and builds the environment in ``build(cfg)``.

This example, ``isaaclab_arena_examples/external_environments/basic.py``, places one object on a
table. Copy it into your package, for example as
``my_package/isaaclab_arena_environments/my_environment.py``:

.. literalinclude:: ../../../isaaclab_arena_examples/external_environments/basic.py
   :language: python
   :start-at: from __future__ import annotations

To run it, pass the factory class to ``--external_environment_class_path``, followed by the
environment name and its flags. For example, with a zero-action policy:

.. code-block:: bash

   python isaaclab_arena/evaluation/policy_runner.py \
     --policy_type zero_action \
     --num_steps 50 \
     --external_environment_class_path my_package.isaaclab_arena_environments.my_environment.ExternalFrankaTableEnvironment \
     franka_table \
     --object tomato_soup_can

The class path is the module path and the class name, joined by a dot as in a Python import. For a
class ``MyEnvironment`` in ``my_package/my_environment.py``, it is
``my_package.my_environment.MyEnvironment``. The module must be importable, for example because your
package is installed or on ``PYTHONPATH``.

.. note::

    The example is not a built-in environment, so it also runs through
    ``--external_environment_class_path`` from the Arena source tree:

    .. code-block:: bash

        python isaaclab_arena/evaluation/policy_runner.py \
          --viz kit \
          --policy_type zero_action \
          --num_steps 50 \
          --external_environment_class_path isaaclab_arena_examples.external_environments.basic.ExternalFrankaTableEnvironment \
          franka_table \
          --object tomato_soup_can

    which results in an environment like the one below:

    .. image:: ../../images/externally_defined_environment.png
       :width: 50%
       :alt: External environment example
       :align: center

Next Steps
----------

The example above uses a built-in task (``NoTask``) and a built-in embodiment
(``franka_ik``).  To learn how to define your own custom tasks and embodiment
variants, see :doc:`external_tasks_and_embodiments`.
