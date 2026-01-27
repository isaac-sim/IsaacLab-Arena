Assets Management
=================

.. note::

These steps are only required if you are adding your own assets to Isaac Lab Arena's production S3 buckets.
Assets and files that are used by Isaac Lab Arena by default are publicly available without additional authentication.


Issac Lab Arena consumes assets from publicly available S3 buckets and LightWheel SDK registry.
Isaac Lab Arena uses the following environment variables to access those S3 buckets:

- ``ISAAC_NUCLEUS_DIR``: The directory of Isaac Sim's assets on the production S3 bucket, containing assets that are compliant with legal requirements.
- ``ISAACLAB_NUCLEUS_DIR``: The directory of Isaac Lab & Isaac Lab Arena's assets on the production S3 bucket, containing assets that are compliant with legal requirements.
- ``ISAACLAB_STAGING_NUCLEUS_DIR``: The directory of Isaac Lab Arena's development assets on the staging S3 bucket, auto-sync every 6 hours
from the Nucleus Server (omniverse://isaac-dev.ov.nvidia.com/Isaac/IsaacLab/Arena/). Those assets are not yet reviewed by legal team, using in production at your own risk.
Prior to each major release, the staging assets will be uploaded to the production bucket after cleared by legal team.

Adding assets
-------------

To add your own assets to Isaac Lab Arena's *staging* S3 bucket, you can follow the following steps:

1. Upload your assets to the Nucleus Server (``omniverse://isaac-dev.ov.nvidia.com/Isaac/IsaacLab/Arena``). Depend on the type of your assets, you can upload them to the following directories:
   - Backgrounds: ``/Isaac/IsaacLab/Arena/assets/background_library``
   - Objects: ``/Isaac/IsaacLab/Arena/assets/object_library``
   - Embodiments: ``/Isaac/IsaacLab/Arena/assets/robot_library``
   - Policies: ``/Isaac/IsaacLab/Arena/${NAME_OF_THE_POLICY}``
Prefix assets with their source (e.g., scarif_tower_01.usd), upload to ``/Isaac/IsaacLab/Arena/assets/object_library``,
and include a valid license file in the target folder. If you lack Nucleus Server write access, contact the isaaclab-arena team to facilitate the asset and license upload.

2. Update the USD path configurations to use the new assets as they are stored in ``ISAACLAB_STAGING_NUCLEUS_DIR/Arena/assets`` directory.
For example, if you want to use a new object, you can register the object in ``isaaclab_arena/assets/object_library.py`` as a derived class of ``LibraryObject``, and add the USD path as its class attribute.

3. Wait for at most 6 hours for the assets to be synced to the staging S3 bucket.

4. Run Isaac Lab Arena to verify that the assets are loaded correctly.

