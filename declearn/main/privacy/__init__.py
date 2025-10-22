# coding: utf-8

# Copyright 2025 Inria (Institut National de Recherche en Informatique
# et Automatique)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DEPRECATED Submodule implementing Differential-Privacy-oriented tools.

This module was moved to `declearn.training.dp` as of DecLearn 2.6, and is
only re-exported for retro-compatibility. It will be removed in DecLearn 2.8.

* [DPTrainingManager][declearn.training.dp.DPTrainingManager]:
    TrainingManager subclass implementing Differential Privacy mechanisms.
"""

# pragma: no cover

import warnings

from declearn.training.dp import DPTrainingManager

warnings.warn(
    "'declearn.main.privacy' was moved to `declearn.training.dp` and is only "
    "re-exported for retro-compatibility. It will be removed in DecLearn 2.8.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["DPTrainingManager"]
