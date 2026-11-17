# coding: utf-8

# Copyright 2026 Inria (Institut National de Recherche en Informatique
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

"""Utils related to general errors."""


class IncompatibilityError(Exception):
    """Custom exception to indicate that incompatible configs/modules/features
    are selected/activated at the same time
    """

    pass


class InvalidDatasetType(TypeError):
    """Custom TypeError to indicate that the received dataset is of
    invalid type
    """

    def __init__(self, target: str, recv_data_type):
        super().__init__(f"{target} must be a Dataset. Got : {recv_data_type}")
