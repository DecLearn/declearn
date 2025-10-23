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

"""Sparse matrix file dumping and loading utils, inspired by svmlight.

The format used is mostly similar to the SVMlight one (see for example
`sklearn.datasets.dump_svmlight_file`), but enables storing a single
matrix rather than a (X, y) pair of arrays. It also records the input
matrix's dtype and type of sparse format, which are thus restored when
reloading - while the scikit-learn implementation always returns a CSR
matrix and requires inputing the dtype.

This implementation does not use any tricks (e.g. cython or interfacing an
external c++ tool) to optimize dump or load runtimes. It may therefore be
slower than using the scikit-learn functions or any third-party alternative.
"""

import json
import os

from scipy.sparse import (  # type: ignore
    bsr_matrix,
    coo_matrix,
    csc_matrix,
    csr_matrix,
    dia_matrix,
    dok_matrix,
    lil_matrix,
    spmatrix,
)

__all__ = [
    "sparse_from_file",
    "sparse_to_file",
]


SPARSE_TYPES = {
    bsr_matrix: "bsr",
    csc_matrix: "csc",
    csr_matrix: "csr",
    coo_matrix: "coo",
    dia_matrix: "dia",
    dok_matrix: "dok",
    lil_matrix: "lil",
}


def sparse_to_file(
    path: str,
    matrix: spmatrix,
) -> None:
    """Dump a scipy sparse matrix as a text file.

    See the [`sparse_from_file`][declearn.dataset.utils.sparse_from_file]
    counterpart function to reload the dumped data from the created file.

    Parameters
    ----------
    path: str
        Path to the file where to store the sparse matrix.
        If the path does not end with a '.sparse' extension,
        one will be added automatically.
    matrix: scipy.sparse.spmatrix
        Sparse matrix that needs storing to file.

    Raises
    ------
    TypeError
        If 'matrix' is of unsupported type, i.e. not a BSR,
        CSC, CSR, COO, DIA, DOK or LIL sparse matrix.

    Note
    ----
    The format used is mostly similar to the SVMlight one (see for example
    `sklearn.datasets.dump_svmlight_file`), but enables storing a single
    matrix rather than a (X, y) pair of arrays. It also records the input
    matrix's dtype and type of sparse format, which are restored when the
    counterpart `sparse_from_file` function is used to reload it.
    """
    if os.path.splitext(path)[1] != ".sparse":
        path += ".sparse"
    # Identify the type of sparse matrix, and convert it to lil.
    name = SPARSE_TYPES.get(type(matrix))  # type: ignore
    if name is None:
        raise TypeError(f"Unsupported sparse matrix type: '{type(matrix)}'.")
    lil = matrix.tolil()  # type: ignore
    # Record key metadata required to rebuild the matrix.
    meta = {
        "stype": name,
        "dtype": lil.dtype.name,
        "shape": lil.shape,
    }
    # Write data to the target file.
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        file.write(json.dumps(meta))
        for ind, val in zip(lil.rows, lil.data, strict=False):
            row = " ".join(f"{i}:{v}" for i, v in zip(ind, val, strict=False))
            file.write("\n" + row)


def sparse_from_file(path: str) -> spmatrix:
    """Return a scipy sparse matrix loaded from a text file.

    See the [`sparse_to_file`][declearn.dataset.utils.sparse_to_file]
    counterpart function to create reloadable sparse data dump files.

    Parameters
    ----------
    path: str
        Path to the sparse matrix dump file.

    Returns
    -------
    matrix: scipy.sparse.spmatrix
        Sparse matrix restored from file, the exact type
        of which being defined by said file.

    Raises
    ------
    KeyError
        If the file's header cannot be JSON-parsed or does not
        conform to the expected standard.
    TypeError
        If the documented sparse matrix type is not supported,
        i.e. "bsr", "csv", "csc", "coo", "dia", "dok" or "lil".


    Note
    ----
    The format used is mostly similar to the SVMlight one (see for example
    `sklearn.datasets.load_svmlight_file`), but the file must store a single
    matrix rather than a (X, y) pair of arrays. It must also record some
    metadata in its header, which are notably used to restore the initial
    matrix's dtype and type of sparse format.
    """
    with open(path, "r", encoding="utf-8") as file:
        # Read and parse the file's header.
        try:
            head = json.loads(file.readline())
        except json.JSONDecodeError as exc:
            raise KeyError("Invalid header for sparse matrix file.") from exc
        if any(key not in head for key in ("stype", "dtype", "shape")):
            raise KeyError("Invalid header for sparse matrix file.")
        if head["stype"] not in SPARSE_TYPES.values():
            raise TypeError(f"Invalid sparse matrix type: '{head['stype']}'.")
        # Instantiate a lil_matrix abiding by the header's specs.
        lil = lil_matrix(tuple(head["shape"]), dtype=head["dtype"])
        cnv = int if lil.dtype.kind == "i" else float
        # Iteratively parse and fill-in row data.
        for rix, row in enumerate(file):
            _row = row.strip(" \n")
            if not _row:  # all-zeros row
                continue
            for field in _row.split(" "):
                ind, val = field.split(":")
                lil[rix, int(ind)] = cnv(val)
    # Convert the matrix to its initial format and return.
    return getattr(lil, f"to{head['stype']}")()
