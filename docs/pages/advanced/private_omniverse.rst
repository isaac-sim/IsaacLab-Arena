Storage API Access
==================

These steps are for NVIDIA employees accessing internal Arena assets. Public S3
assets do not require login. Use ``omni.client`` 2.74.0 or newer.

1. Request write access (optional)
----------------------------------

This step is only needed to upload or modify assets. Join the
``access-isaac-dev-storage-arena-write`` group on
`DLRequest <https://dlrequest/GroupID/Home/Index>`__.

2. Sign in and verify access
----------------------------

Run the setup script from the repository root in your installed Arena runtime.
It preserves existing credentials, starts NVIDIA device login only when credentials
are missing, then lists the Arena storage root and reports whether you have write
access.

The script implements the device-code flow from the Storage API
`authentication instructions <https://gitlab-master.nvidia.com/omniverse/usd-storage/ops/sapi-deployments/-/blob/master/doc/user/auth/how-to.md>`__
and the credential-file setup from the
`login quickstart <https://gitlab-master.nvidia.com/omniverse/usd-storage/ops/sapi-deployments/-/blob/master/doc/user/kit/kit-vm-quickstart.md>`__.

.. tab-set::

   .. tab-item:: Native uv
      :selected:

      .. code-block:: bash

         uv run python tools/internal/setup_storage_access.py

   .. tab-item:: Docker

      Run inside the Arena container:

      .. code-block:: bash

         python tools/internal/setup_storage_access.py

Credentials are saved at ``~/.nvidia-omniverse/config/auth.toml`` with mode
``600``. Host and container credentials are separate because each runtime uses
its own home directory. The client refreshes access tokens automatically. Keep
this file private.

For renewing an existing login, follow the internal
`device-login guide <https://gitlab-master.nvidia.com/omniverse/usd-storage/ops/sapi-deployments/-/blob/master/doc/user/kit/kit-vm-quickstart.md>`__.
For unattended CI, use the
`Storage API federation guide <https://gitlab-master.nvidia.com/omniverse/usd-storage/ops/sapi-deployments/-/blob/master/doc/user/auth/federation.md>`__.
