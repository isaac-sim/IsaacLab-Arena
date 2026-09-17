Several robots sharing policies
================================

Use the registered composite policy to drive several robot instances in one
environment. It maps each robot's scene key to a named inner policy instance.
Robots assigned to the same instance share its weights and receive one batched
policy call per environment step.

Configure the composition in the ordinary flat experiment format:

.. code-block:: yaml

   policy:
     type: multi_robot
     policies:
       arms:
         type: zero_action
         params: {}
       humanoid:
         type: zero_action
     assignments:
       left: arms
       right: arms
       robot: humanoid

An unkeyed embodiment keeps the scene key ``robot``.
See :doc:`../embodiment/index` for the naming rule.

The environment must contain the corresponding robots. Inner policies use
registered type names and primitive parameters. Nested composite policies are
not supported.

Shared robots must have matching observation groups, camera terms, action names,
action widths, and action bounds after removing their instance prefixes.
The composite stacks observations in robot order, then parallel-environment order.
For two robots and three parallel environments, rows zero through two belong to
the first robot; rows three through five belong to the second.
It scatters returned action columns back into the full environment tensor.

The environment view passed to an inner policy exposes its action manager,
action spaces, batch size, and device. It does not expose the entire scene.
The RSL-RL reinforcement-learning wrapper expects additional environment methods,
including reset. The current view does not provide them. Checkpoint integration
has not been tested, and that wrapper is not supported by this view.

Indexed resets expand to every corresponding robot row. Task descriptions and
shutdown calls reach each inner policy once. Inner policies must agree on whether
they have a recorded horizon and on its length. The composite reports itself as
remote when any inner policy is remote; it does not add a remote shutdown method
that the base policy interface does not define.
