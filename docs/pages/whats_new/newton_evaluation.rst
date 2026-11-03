Newton-Based Evaluation Workflows
=================================

Arena environments can run on the Newton physics backend as well as PhysX. The upcoming release expands
Newton support for objects and robots, so contact-rich and deformable tasks can be previewed,
teleoperated, recorded, and evaluated with the same Arena workflows.

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: Deformables
      :link: ../example_workflows/newton/deformable_pick_and_place
      :link-type: doc

      .. video:: ../../images/whats_new/arena_v04_newton_deformables.mp4
         :width: 100%
         :alt: DROID placing a deformable block on a plate, shown as the rendered scene and the Newton deformable mesh
         :autoplay:
         :loop:
         :muted:
         :playsinline:
         :nocontrols:

      Build evaluation environments with objects that deform during interaction.

   .. grid-item-card:: Contact-Rich Insertion
      :link: ../example_workflows/newton/gear_insertion
      :link-type: doc

      .. video:: ../../images/whats_new/arena_v04_newton_rigid_gear_insertion.mp4
         :width: 100%
         :alt: DROID inserting a gear onto a peg, shown as the rendered scene and the Newton collision geometry
         :autoplay:
         :loop:
         :muted:
         :playsinline:
         :nocontrols:

      Build evaluation environments for tasks requiring tight-tolerance insertions.
