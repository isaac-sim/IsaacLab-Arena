# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Eagle config compat: alias _attn_implementation_autoset and set flash_attention_2 on loaded configs."""

import contextlib

_APPLIED = False


def apply_eagle_config_compat() -> None:
    """Apply once per process. Idempotent."""
    global _APPLIED
    if _APPLIED:
        return
    import transformers
    import transformers.configuration_utils as configuration_utils

    _orig_getattribute = configuration_utils.PretrainedConfig.__getattribute__

    def _compat_getattribute(self, name: str):
        if name == "_attn_implementation_autoset":
            with contextlib.suppress(AttributeError):
                return _orig_getattribute(self, "_attn_implementation_internal")
        return _orig_getattribute(self, name)

    configuration_utils.PretrainedConfig.__getattribute__ = _compat_getattribute

    _orig_from_pretrained = transformers.AutoConfig.from_pretrained

    @classmethod
    def _wrapped_from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        config = _orig_from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        for sub in ("text_config", "vision_config"):
            sub_config = getattr(config, sub, None)
            if sub_config is not None and getattr(sub_config, "_attn_implementation", None) != "flash_attention_2":
                sub_config._attn_implementation = "flash_attention_2"
        return config

    transformers.AutoConfig.from_pretrained = _wrapped_from_pretrained
    _APPLIED = True
