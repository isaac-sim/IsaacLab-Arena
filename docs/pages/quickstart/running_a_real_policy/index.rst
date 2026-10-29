Running a Real Policy
=====================

The earlier zero-action example keeps the robot still, so its success rate stays at zero. In this
section, you replace that policy with a pre-trained model. Each guide shows how to start the model
server and run a saved YAML configuration.

Start with OpenPI for an interactive first rollout, then try GR00T. The VLM agent guide uses a
hosted vision-language model as the policy, so it needs an inference API key instead of a model
server.

.. toctree::
   :maxdepth: 1

   openpi
   gr00t
   vlm_agent
