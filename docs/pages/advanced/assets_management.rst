Assets Management
=================

.. note::

   **Internal NVIDIA use only.** This page documents workflows and infrastructure (Storage API,
   staging/production S3 buckets, internal ticketing) that are only accessible to NVIDIA employees.

.. note::

   These steps are only required if you are adding your own assets to Isaac Lab Arena's production S3 buckets.
   Assets and files that are used by Isaac Lab Arena by default are publicly available without additional authentication.


Isaac Lab Arena consumes assets from publicly available S3 buckets and LightWheel SDK registry.
Isaac Lab Arena uses the following constants to access those assets:

- ``ISAAC_NUCLEUS_DIR``: The directory of Isaac Sim's assets on the production S3 bucket, containing assets that are compliant with legal requirements.
- ``ISAACLAB_NUCLEUS_DIR``: The directory of Isaac Lab & Isaac Lab Arena's assets on the production S3 bucket, containing assets that are compliant with legal requirements.
- ``ARENA_STAGING_STORAGE_DIR``: The directory of Isaac Lab Arena's internal development assets in Storage API
  (``ovstorage://isaac/Isaac/IsaacLab``). Those assets have not yet been reviewed by the legal team -- use in production at your own risk.
  Prior to each major release, the staging assets will be uploaded to the production bucket after being cleared by the legal team.

.. note::

   In Isaac Lab 3.0-alpha, ``ISAACLAB_NUCLEUS_DIR`` currently points to the *staging* S3 bucket rather than the production bucket.
   See the `Isaac Lab staging-bucket configuration <https://github.com/isaac-sim/IsaacLab/blob/55df2c34390ba94b22d41879514c5485c5115462/apps/isaaclab.python.kit#L309>`__.

Adding assets
-------------

Before uploading, authenticate with the Storage API by following :doc:`private_omniverse`.

For immediate access to new uploads, use ``ARENA_STAGING_STORAGE_DIR`` from
``isaaclab_arena.assets.nucleus``. It points to the internal Storage API location
``ovstorage://isaac/Isaac/IsaacLab`` and requires internal access.
Append ``/Arena/assets/...`` to build an asset path. ``ARENA_NUCLEUS_DIR`` uses the
public S3 mirror, where uploads become available after synchronization.

To add your own assets to Isaac Lab Arena's *staging* S3 bucket, you can follow the following steps:

1. Open `Storage Navigator <https://navigator.rc.storage.omni.nvidia.com/https%3A%2F%2Fstorage.rc.storage.omni.nvidia.com/path/https%3A%2F%2Fstsapiwus3isaac.blob.core.windows.net%2Fisaac%2FIsaac%2FIsaacLab%2FArena%2F>`__
   and upload your assets to the Storage API. Depending on the type of your assets, you can upload them to the following directories:

   - Backgrounds: ``/Isaac/IsaacLab/Arena/assets/background_library``
   - Objects: ``/Isaac/IsaacLab/Arena/assets/object_library``
   - Embodiments: ``/Isaac/IsaacLab/Arena/assets/robot_library``
   - Policies: ``/Isaac/IsaacLab/Arena/${NAME_OF_THE_POLICY}``

   Prefix assets with their source (e.g., ``scarif_tower_01.usd``), upload to ``/Isaac/IsaacLab/Arena/assets/object_library``,
   and include a valid license file in the target folder. If you lack Storage API write access, contact the isaaclab-arena team to facilitate the asset and license upload.

2. Update the USD path configurations to use the new assets as they are stored in ``ARENA_STAGING_STORAGE_DIR/Arena/assets``.
   For example, if you want to use a new object, you can register the object in ``isaaclab_arena/assets/object_library.py`` as a derived class of ``LibraryObject``, and add the USD path as its class attribute.

3. Request synchronization to the staging S3 bucket on `ISIM-5044 <https://nvidia.atlassian.net/browse/ISIM-5044>`__.

4. Run Isaac Lab Arena to verify that the assets are loaded correctly.
