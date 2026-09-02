.. _internal-osmo-prerequisites:

Internal OSMO Prerequisites
===========================

The :doc:`multi-node evaluation example
<../example_workflows/multi_node_evaluation/multi_node_evaluation>` uses NVIDIA-internal
OSMO, private NGC images, and CSS output storage. Follow the `OSMO 6.3 User Guide
<https://nvidia.github.io/OSMO/release/6.3/user_guide/index.html>`_ for client and
profile setup, then apply the Arena-specific settings below.

Internal service and compute
----------------------------

Request membership in the `access-osmo group <http://nv/access-osmo>`_ if needed,
then log in with your NVIDIA account:

.. code-block:: bash

   osmo login https://us-west-2-aws.osmo.nvidia.com --method code

Login does not create workflow credentials.

Use ``osmo pool list`` and ``osmo resource list --pool isaac-dev-l40-03`` to find
a pool and platform available to your account.

NGC registry credential
-----------------------

The workflow pulls from ``nvcr.io/nvstaging/isaac-amr``. Request Registry User
membership in the `access-isaac-ngc-user group
<https://dlrequest/GroupID/Groups/Properties?identity=NTZhNDE2ZmE2NmQyNDIwYTljYWFiMjgxZDJhMmM4MDV8Z3JvdXA=>`_,
accept the ``nvstaging`` invitation if required, then follow the OSMO `NGC registry
credential instructions
<https://nvidia.github.io/OSMO/release/6.3/user_guide/getting_started/credentials.html#credentials-registry>`_.
The resulting credential profile must be ``nvcr.io``.

Omniverse credential
--------------------

Arena requires a ``GENERIC`` credential named ``omni_svc``. Create an API token
for ``isaac-dev.ov.nvidia.com`` using the `Omniverse token instructions
<https://docs.omniverse.nvidia.com/nucleus/latest/config-and-info/api_tokens.html>`_,
then register it:

.. code-block:: bash

   osmo credential set omni_svc \
     --type GENERIC \
     --payload \
       'omni_user=$omni-api-token' \
       'omni_pass=<OMNIVERSE_API_TOKEN>'

.. _internal-osmo-css-data-credential:

CSS data credential
-------------------

Follow the internal `CSS data credential instructions
<https://isaac-infrastructure.gitlab-master-pages.nvidia.com/osmo/main/user_guide/appendix/css/index.html#data-credentials-css>`_
to request access and obtain the S3 ACL access user and secret key. Register the
shared Arena account at the account level:

.. code-block:: bash

   osmo credential set team-isaac-data \
     --type DATA \
     --payload \
       endpoint=swift://pdx.s8k.io/AUTH_team-isaac \
       region=us-east-1 \
       'access_key_id=<S3_ACL_ACCESS_USER>' \
       'access_key=<S3_SECRET_KEY>'

Use the corresponding output URL:

.. code-block:: text

   swift://pdx.s8k.io/AUTH_team-isaac/isaaclab_arena/workflows/{{workflow_id}}

Check the ``isaaclab_arena`` container before submitting:

.. code-block:: bash

   osmo data check \
     swift://pdx.s8k.io/AUTH_team-isaac/isaaclab_arena \
     --access-type WRITE

``osmo credential list`` should show a ``REGISTRY`` profile for ``nvcr.io``, the
``omni_svc`` credential, and a ``DATA`` profile matching the output account. The
DATA credential must show ``Local Yes`` for ``osmo data check``.

SQA output account
~~~~~~~~~~~~~~~~~~

For SQA, ``AUTH_team-isaac-sqa`` is the account and ``isaac-sqa`` is the
container. Register the SQA keys with
``endpoint=swift://pdx.s8k.io/AUTH_team-isaac-sqa`` and use:

.. code-block:: text

   swift://pdx.s8k.io/AUTH_team-isaac-sqa/isaac-sqa/isaaclab_arena/workflows/{{workflow_id}}

Check the container with:

.. code-block:: bash

   osmo data check \
     swift://pdx.s8k.io/AUTH_team-isaac-sqa/isaac-sqa \
     --access-type WRITE

A ``HeadBucket`` 404 means that the credential cannot see the ``isaac-sqa``
container or that its name is wrong. Confirm the container and its ACL in CSS;
the later ``isaaclab_arena/workflows/{{workflow_id}}`` prefix is not part of the
container name.
