# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Wrapper around an ONNX Runtime inference session."""

import numpy as np

import onnxruntime as ort


class OnnxInferenceSession:
    """Thin wrapper around onnxruntime.InferenceSession with named-dict I/O."""

    def __init__(self, model_path: str):
        """Load an ONNX model and record its input and output names.

        Args:
            model_path: Path to the ONNX model file.
        """
        self.session = ort.InferenceSession(model_path)
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
        print(f"Successfully loaded ONNX policy from {model_path}")

    def run(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Run inference and return outputs as a named dict.

        Args:
            inputs: Mapping from model input name to input array.

        Returns:
            Mapping from model output name to output array.
        """
        outputs = self.session.run(self.output_names, inputs)
        return dict(zip(self.output_names, outputs))
