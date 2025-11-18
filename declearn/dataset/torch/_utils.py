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

"""Utils to handle torch datasets and power up declearn's `TorchDataset`."""

from typing import List, Optional, Tuple, Union

import torch

__all__ = [
    "PoissonSampler",
    "collate_with_padding",
]


class PoissonSampler(torch.utils.data.Sampler):
    """Custom `torch.utils.data.Sampler` implementing Poisson sampling.

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
        # super init is empty and its signature will change in torch 2.2
        # pylint: disable=super-init-not-called
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


def collate_with_padding(
    samples: List[Tuple[Union[torch.Tensor, List[torch.Tensor]], ...]],
) -> Tuple[Union[List[torch.Tensor], torch.Tensor], ...]:
    """Collate input elements into batches, with padding when required.

    This custom collate function is designed to enable padding samples of
    variable length as part of their stacking into mini-batches. It relies
    on the `torch.nn.utils.rnn.pad_sequence` utility function and supports
    receiving samples that contain both inputs that need padding and that
    do not (e.g. variable-length token sequences as inputs but fixed-size
    values as labels).

    It may be used as `collate_fn` argument to the declearn `TorchDataset`
    to wrap up data that needs such collation - but users are free to set
    up and use their own custom function if this fails to fit their data.

    Parameters
    ----------
    samples:
        Sample-level records, formatted as (same-structure) tuples with
        torch tensors and/or lists of tensors as elements. None elements
        are also supported.

    Returns
    -------
    batch:
        Tuple with the same structure as input ones, collating sample-level
        records into batched tensors.
    """
    output: List[Union[List[torch.Tensor], torch.Tensor]] = []
    for i, element in enumerate(samples[0]):
        if element is None:
            output.append(None)
            continue
        if isinstance(element, (list, tuple)):
            out: Union[torch.Tensor, List[torch.Tensor]] = [
                torch.nn.utils.rnn.pad_sequence(
                    [smp[i][j] for smp in samples],
                    batch_first=True,
                )
                for j in range(len(element))
            ]
        elif element.shape:
            out = torch.nn.utils.rnn.pad_sequence(
                [smp[i] for smp in samples],  # type: ignore  # false-positive
                batch_first=True,
            )
        else:
            out = torch.stack(  # pylint: disable=no-member
                [smp[i] for smp in samples]  # type: ignore
            )
        output.append(out)
    return tuple(output)
