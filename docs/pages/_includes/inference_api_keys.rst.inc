.. Shared inference API key setup, included by every page whose workflow calls an inference endpoint.

Export one or more API keys
in the host environment **before** launching a native ``uv`` runner or starting the
Docker container. Docker forwards the configured keys when it creates the
container. This step is required only once per host environment.

.. tab-set::

   .. tab-item:: NVIDIA Public Endpoint
      :selected:

      Generate an NGC API key at
      `build.nvidia.com API keys <https://build.nvidia.com/settings/api-keys>`_,
      then export it for the publicly reachable endpoint:

      .. code-block:: bash

         export NVIDIA_API_KEY="YOUR_NGC_API_KEY"

   .. tab-item:: NVIDIA Internal Endpoint

      From the NVIDIA network, generate an internal API key at
      `inference.nvidia.com key management <https://inference.nvidia.com/key-management>`_,
      then export it:

      .. code-block:: bash

         export NV_API_KEY="YOUR_INTERNAL_API_KEY"

      .. note::

         This endpoint is accessible by NVIDIA employees only and counts into your Inference Hub token usage.

   .. tab-item:: OpenAI Endpoint

      Create an OpenAI account, generate a secret at
      `OpenAI API keys <https://platform.openai.com/api-keys>`_, and configure
      payment or prepaid credits under
      `OpenAI billing <https://platform.openai.com/settings/organization/billing/overview>`_.
      OpenAI charges the account associated with the key according to its
      current API pricing and usage. Store the key securely and do not commit it
      to the repository.

      .. code-block:: bash

         export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"

      .. note::

         The ``openai`` endpoint connects directly to a third-party service
         operated by OpenAI, not NVIDIA. Its availability, regional
         restrictions, data handling, pricing, and terms are controlled by
         OpenAI. Review those terms before sending prompts or other data.

Set ``ARENA_INFERENCE_ENDPOINT`` to choose the default endpoint:

.. code-block:: bash

   export ARENA_INFERENCE_ENDPOINT=public  # internal, public, or openai
