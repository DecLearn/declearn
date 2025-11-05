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

from typing import Any, Dict

import pandas as pd
import torch
from transformers import PreTrainedTokenizer


class TextDataset(torch.utils.data.Dataset):
    """
    Custom torch Dataset for our example
    We retrieve text and label data from a dataframe,
    tokenize and provide ( (input_ids, attention_mask), label )
    as output when the dataset is queried

    Note: Here, for simplicity purpose, we perform tokenization during item
    retrieval. In a real application, it may be more efficient to tokenize
    the entire dataset before, store the result and only retrieve tokenized
    data when invoking __getitem__
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer: PreTrainedTokenizer,
        text_field: str = "text",
        label_field: str = "label",
        max_length: int = 512,
    ) -> None:
        self.texts = dataframe[text_field].tolist()
        self.labels = dataframe[label_field].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)  # shape: [max_length]
        attention_mask = encoding["attention_mask"].squeeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return (input_ids, attention_mask), label
