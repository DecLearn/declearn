# coding: utf-8

# Copyright 2023 Inria (Institut National de Recherche en Informatique
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

"""Backend utils for 'TorchDataset'."""

from typing import Optional

import torch

__all__ = [
    "PoissonSampler",
]


class PoissonSampler(torch.utils.data.Sampler):
    """Custom 'torch.utils.data.Sampler' implementing Poisson sampling.

    This sampler is equivalent to the `UniformWithReplacementSampler`
    from the third-party `opacus` library, with the exception that it
    skips empty batches, preventing issues at collate time.
    """

    def __init__(
        self,
        num_samples: int,
        sample_rate: float,
        generator: Optional[torch.Generator] = None,
        # false-positive on 'torch.Generator'; pylint: disable=no-member
    ) -> None:
        """Instantiate a Poisson (UniformWithReplacement) Sampler.

        Parameters
        ----------
        num_samples: int
            Number of samples in the dataset to sample from.
        sample_rate: float
            Sampling rate, i.e. probability for each sample to be included
            in any given batch. Hence, average number of samples per batch.
        generator: torch.Generator or None, default=None
            Optional RNG, that may be used to produce seeded results.
        """
        self.num_samples = num_samples
        self.sample_rate = sample_rate
        self.generator = generator

    def __len__(self):
        return int(1 / self.sample_rate)

    def __iter__(self):
        for _ in range(len(self)):
            # Draw a random batch of samples based on Poisson sampling.
            rand = torch.rand(  # false-positive; pylint: disable=no-member
                self.num_samples, generator=self.generator, device="cpu"
            )
            indx = (rand < self.sample_rate).nonzero().reshape(-1).tolist()
            # Yield selected indices, unless the batch would be empty.
            if not indx:
                continue
            yield indx
