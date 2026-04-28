Omniverse Authentication
========================

.. note::

    These steps are only required if you are extending Isaac Lab Arena with your own
    internal assets. Assets and files that are used by Isaac Lab Arena by default are
    publicly available without additional authentication (i.e. without performing the
    steps detailed on this page).

To allow the project to access Nvidia Omniverse assets or services
e.g., stored scenes, USD files, or extensions on a **private** Nucleus server,
you must authenticate using an Omniverse API token.

Isaac Sim and other Nvidia Omniverse applications use two environment variables
``OMNI_USER`` and ``OMNI_PASS`` to authenticate automatically.

To generate those, follow the instructions on the
`Omniverse Authentication documentation <https://docs.omniverse.nvidia.com/nucleus/latest/config-and-info/api_tokens.html>`_.

.. code-block:: bash

    export OMNI_USER='$omni-api-token'
    export OMNI_PASS=<your_generated_api_token>

Where:

* ``OMNI_USER`` should be set to the literal string ``$omni-api-token`` (including the leading dollar sign)

* ``OMNI_PASS`` should be the actual token value you generated.
