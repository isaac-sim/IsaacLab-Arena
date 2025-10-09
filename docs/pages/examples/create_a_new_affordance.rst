Creating a New Affordance
=========================

This guide explains how to create a new affordance using the ``Openable`` affordance as an example.

Overview
--------

Affordances define interactive behaviors for objects in Isaac Arena. They encapsulate joint-based interactions like opening doors, pressing buttons, or rotating knobs.

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
   - Store joint name and behavioral parameters
   - Accept threshold values for state detection

**2. State Query Methods**
   - Check current state (e.g., ``is_open()``, ``get_openness()``)
   - Use ``get_normalized_joint_position()`` for joint readings
   - Return boolean tensors for multi-environment support

**3. Action Methods**
   - Manipulate object state (e.g., ``open()``, ``close()``)
   - Use ``set_normalized_joint_position()`` to control joints
   - Support partial actions via percentage parameters

**4. Helper Methods**
   - ``_add_joint_name_to_scene_entity_cfg()`` configures scene entities
   - Handle default ``SceneEntityCfg`` creation when not provided

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
   - Always use normalized joint positions (0.0 to 1.0 range)
   - Configure ``asset_cfg.joint_names`` before joint operations
   - Handle multi-environment scenarios with ``env_ids``

**Threshold Flexibility**
   - Allow runtime threshold overrides in query methods
   - Store default thresholds as instance variables
   - Use sensible defaults (typically 0.5)

**Error Handling**
   - Provide default ``SceneEntityCfg`` when None
   - Use object's ``name`` property for asset identification
   - Support optional parameters for flexibility

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

This pattern enables rich, interactive object behaviors while maintaining clean separation of concerns.
