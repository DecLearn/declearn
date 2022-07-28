# coding: utf-8

"""Type hinting utils, defined and exposed for code readability purposes."""

from typing import List, Optional, Tuple, Union

from numpy.typing import ArrayLike


__all__ = [
    'Batch',
]

# Data batches specification: (inputs, labels, weights), where:
# - inputs and labels may be an array or a list of arrays:
# - labels and/or weights may ne None
Batch = Tuple[
    Union[ArrayLike, List[ArrayLike]],
    Optional[Union[ArrayLike, List[ArrayLike]]],
    Optional[ArrayLike]
]
