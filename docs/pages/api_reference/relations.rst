Relations Reference
===================

The page documents the spatial relations available in Isaac Lab - Arena.
For the placement checks named in each *Validation* section, see :doc:`/pages/concepts/object_placement/validation`.

.. currentmodule:: isaaclab_arena.relations.relations

Summary
-------

.. autosummary::
   :nosignatures:

   AlignWith
   AtPosition
   ClutterOn
   FaceTo
   IsAnchor
   NextTo
   NotNextTo
   On
   PositionLimitsBox
   PositionLimitsCylindrical
   RequiresReachability
   RotateAroundSolution

API reference
-------------

.. autoclass:: AlignWith
.. autoclass:: AtPosition
.. autoclass:: ClutterOn
.. autoclass:: FaceTo
.. autoclass:: IsAnchor
.. autoclass:: NextTo
.. autoclass:: NotNextTo
.. autoclass:: On
.. autoclass:: PositionLimitsBox
.. autoclass:: PositionLimitsCylindrical
.. autoclass:: RequiresReachability
.. autoclass:: RotateAroundSolution

Enum parameters take their string values, for example ``side: positive_x``.

.. autoclass:: Side
   :members:
   :undoc-members:
   :member-order: bysource

.. autoclass:: isaaclab_arena.relations.axis.Axis
   :members:
   :undoc-members:
   :member-order: bysource
   :exclude-members: index

.. py:class:: PlaceableAsset
   :module: isaaclab_arena.relations.placement_asset

   An asset that relations can place.
