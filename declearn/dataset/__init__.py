# coding: utf-8

"""Dataset-interface API and actual implementations module."""

from ._base import Dataset, DataSpecs, load_dataset_from_json
#from ._sparse import sparse_from_file, sparse_to_file
from ._inmemory import InMemoryDataset
