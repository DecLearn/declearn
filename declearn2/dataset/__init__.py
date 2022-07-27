# coding: utf-8

"""Dataset-interface API and actual implementations module."""

from ._base import Dataset, DataSpecs
#from ._sparse import sparse_from_file, sparse_to_file
from ._sklearn import SklearnDataset
