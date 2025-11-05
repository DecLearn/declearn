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

from typing import List

import torch
import torch.nn.functional as F
from transformers import DistilBertForSequenceClassification


class CustomDistilBertClassif(torch.nn.Module):
    """
    Custom wrapper around HuggingFace model
    It aims to correctly match inputs and outputs of the DistilBERT
    model provided by HuggingFace to other resources, e.g.
    Declearn TorchDataset & Model API

    Here, it means retrieving input_ids and attention_mask as inputs
    and provide only logits as output
    """

    def __init__(self):
        super().__init__()
        self.hf_model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased"
        )

    def forward(self, *inputs: List[torch.tensor]) -> torch.tensor:
        if len(inputs) != 2:
            raise ValueError(
                "This module needs a tuple of 2 tensors as inputs,"
                f"you have provided {len(inputs)} inputs"
            )
        input_ids, attention_mask = inputs
        seq_classif_out = self.hf_model(
            input_ids=input_ids, attention_mask=attention_mask
        )
        return seq_classif_out.logits
