Creating a New Affordance
=========================

Learn how to create affordances using the ``Openable`` affordance as an example.

Overview
--------

Affordances add interactive behaviors to objects - opening doors, pressing buttons, rotating knobs.

Basic Structure
---------------

All affordances inherit from ``AffordanceBase`` and follow this pattern:

.. code-block:: python

    from isaac_arena.affordances.affordance_base import AffordanceBase
    from isaac_arena.utils.joint_utils import get_normalized_joint_position, set_normalized_joint_position

    class MyAffordance(AffordanceBase):
        def __init__(self, joint_name: str, threshold: float = 0.5, **kwargs):
            super().__init__(**kwargs)
            self.joint_name = joint_name
            self.threshold = threshold

Key Components
--------------

**1. Initialization**
   - Store joint name and thresholds
   - Accept parameters for state detection

**2. State Query Methods**
   - Check current state (``is_open()``, ``get_openness()``)
   - Use ``get_normalized_joint_position()`` for joint readings
   - Return boolean tensors for multi-environment support

**3. Action Methods**
   - Change object state (``open()``, ``close()``)
   - Use ``set_normalized_joint_position()`` to control joints
   - Support partial actions with percentage parameters

**4. Helper Methods**
   - ``_add_joint_name_to_scene_entity_cfg()`` configures scene entities
   - Handle default ``SceneEntityCfg`` when not provided

Example: Openable Affordance
-----------------------------

.. code-block:: python

    class Openable(AffordanceBase):
        def __init__(self, openable_joint_name: str, openable_open_threshold: float = 0.5, **kwargs):
            super().__init__(**kwargs)
            self.openable_joint_name = openable_joint_name
            self.openable_open_threshold = openable_open_threshold

        def is_open(self, env: ManagerBasedEnv, asset_cfg: SceneEntityCfg | None = None,
                   threshold: float | None = None) -> torch.Tensor:
            """Check if object is open based on joint position."""
            openness = self.get_openness(env, asset_cfg)
            threshold = threshold or self.openable_open_threshold
            return openness > threshold

        def open(self, env: ManagerBasedEnv, env_ids: torch.Tensor | None,
                asset_cfg: SceneEntityCfg | None = None, percentage: float = 1.0):
            """Open the object to specified percentage."""
            if asset_cfg is None:
                asset_cfg = SceneEntityCfg(self.name)
            asset_cfg = self._add_joint_name_to_scene_entity_cfg(asset_cfg)
            set_normalized_joint_position(env, asset_cfg, percentage, env_ids)

Implementation Tips
-------------------

**Joint Management**
   - Use normalized joint positions (0.0 to 1.0)
   - Configure ``asset_cfg.joint_names`` before operations
   - Handle multi-environment scenarios with ``env_ids``

**Threshold Flexibility**
   - Allow runtime threshold overrides
   - Store defaults as instance variables
   - Use sensible defaults (typically 0.5)

**Error Handling**
   - Provide default ``SceneEntityCfg`` when None
   - Use object's ``name`` property for identification
   - Support optional parameters

Usage in Assets
---------------

Combine affordances with assets using multiple inheritance:

.. code-block:: python

    class Door(Asset, Openable):
        def __init__(self, **kwargs):
            super().__init__(
                openable_joint_name="door_hinge",
                openable_open_threshold=0.8,
                **kwargs
            )
