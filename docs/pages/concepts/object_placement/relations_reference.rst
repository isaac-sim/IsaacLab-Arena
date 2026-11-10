Relations Reference
===================

Every relation an environment graph YAML can use, generated from the relation classes in
``isaaclab_arena/relations/relations.py``. The YAML kind is the name in a YAML ``relations`` entry.
A binary relation takes a ``reference`` object, which the class receives as ``parent``; a unary one
does not. The remaining constructor arguments go under ``params``. The validation section names
the pre-physics check that rejects layouts breaking the relation; a relation without one is only a
solver objective, so valid layouts can miss it slightly. Every layout is also checked for overlaps
by ``no_overlap``. For how to combine relations, see :doc:`relations`.

.. code-block:: yaml

   relations:
     - kind: next_to
       subject: <object>
       reference: <object>
       params:
         side: positive_x

.. currentmodule:: isaaclab_arena.relations.relations

Relations
---------

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

Parameter types
---------------

In YAML, give an enum parameter as its string value, for example ``side: positive_x``.

.. autoclass:: Side
   :members:
   :undoc-members:
   :member-order: bysource

.. autoclass:: isaaclab_arena.relations.axis.Axis
   :members:
   :undoc-members:
   :member-order: bysource
   :exclude-members: index
