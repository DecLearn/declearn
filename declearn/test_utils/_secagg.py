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


"""Routine to set up some SecAgg controllers."""

import secrets
from typing import List, Tuple

from declearn.secagg.masking import MaskingDecrypter, MaskingEncrypter

__all__ = [
    "build_secagg_controllers",
]


def build_secagg_controllers(
    n_peers: int,
) -> Tuple[MaskingDecrypter, List[MaskingEncrypter]]:
    """Setup aligned masking-based encrypters and decrypter.

    Parameters
    ----------
    n_peers:
        Number of clients for which to set up an encrypter.

    Returns
    -------
    decrypter:
        `MaskingDecrypter` instance.
    encrypters:
        List of `MaskingEncrypter` instances with compatible seeds.
    """
    n_pairs = int(n_peers * (n_peers - 1) / 2)
    s_keys = [secrets.randbits(32) for _ in range(n_pairs)]
    clients: List[MaskingEncrypter] = []
    starts = [n_peers - i - 1 for i in range(n_peers)]
    starts = [sum(starts[:i]) for i in range(n_peers)]
    for idx in range(n_peers):
        pos = s_keys[starts[idx] : starts[idx] + n_peers - idx - 1]
        neg = [s_keys[starts[j] + idx - j - 1] for j in range(idx)]
        clients.append(
            MaskingEncrypter(pos_masks_seeds=pos, neg_masks_seeds=neg)
        )
    server = MaskingDecrypter(n_peers=n_peers)
    return server, clients
