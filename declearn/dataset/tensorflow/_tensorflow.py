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

"""Dataset subclass to wrap up 'tensorflow.data.Dataset' instances."""

import dataclasses
import warnings
from typing import (  # fmt: off
    Callable,
    Iterator,
    List,
    Literal,
    Optional,
    Set,
    Tuple,
    Union,
)

import numpy as np
import tensorflow as tf  # type: ignore

from declearn.dataset._base import Dataset, DataSpecs
from declearn.typing import Batch
from declearn.utils import register_type

__all__ = [
    "TensorflowDataset",
]


BatchMode = Literal["default", "padded", "ragged"]


@register_type(group="Dataset")
class TensorflowDataset(Dataset):
    """Dataset subclass to wrap up 'tensorflow.data.Dataset' instances."""

    def __init__(
        self,
        dataset: tf.data.Dataset,
        buffer_size: Optional[int] = None,
        batch_mode: BatchMode = "default",
        seed: Optional[int] = None,
    ) -> None:
        """Wrap up a 'tensorflow.data.Dataset' into a declearn Dataset.

        Parameters
        ----------
        dataset: tensorflow.data.Dataset
            A tensorflow Dataset instance to be wrapped for declearn use.
            The dataset is expected to yield sample-level records, made
            of one to three (tuples of) tensorflow tensors: model inputs,
            target labels and/or sample weights.
        buffer_size: int or None, default=None
            Optional buffer size denoting the number of samples to pre-fetch
            and shuffle when sampling from the original dataset. The higher,
            the better the shuffling, but also the more memory costly.
            If None, use context-based `batch_size * 10` value.
        batch_mode: str in {"default", "padded", "ragged"}
            Flag specifying how to batch inputs. Use "padded" or "ragged" to
            batch up variable-dimension samples (e.g. sequences of tokens),
            using either `tf.data.Dataset.padded_batch` or `.ragged_batch`.
        seed: int or None, default=None
            Optional seed for the random number generator based on which
            the dataset is (optionally) shuffled when generating batches.
            Note that successive batch-generating calls will not yield
            the same results, as the seeded state is not reset on each
            call.

        Notes
        -----
        The wrapped `tensorflow.data.Dataset`:

        - *must* have a fixed length (with TensorFlow <2.13) / *should*
          have an established `cardinality` (TensorFlow >=2.13).
        - should return sample-level (unbatched) elements, as either:
            - (inputs,)
            - (inputs, labels)
            - (inputs, labels, weights)
          where each element may be a (nested structure of) tensor(s).
        - when using `declearn.model.tensorflow.TensorflowModel`:
            - inputs may be a single tensor or list of tensors
            - labels may be a single tensor or None (usually, not None)
            - weights may be a single tensor or None
        """
        # Assign the dataset, parse and validate its specifications.
        self.dataset = dataset
        self._dspecs = parse_and_validate_tensorflow_dataset(self.dataset)
        warn_if_dataset_is_likely_batched(self.dataset)
        # Assign additional parameters and set up an opt.-seeded RNG.
        self.batch_mode = batch_mode
        self.buffer_size = buffer_size
        self.rng = np.random.default_rng(seed)

    def get_data_specs(
        self,
    ) -> DataSpecs:
        return DataSpecs(
            n_samples=self._dspecs.n_samples,
            features_shape=self._dspecs.input_shp,
            classes=self._dspecs.y_classes,
            data_type=self._dspecs.data_type,
        )

    # pylint: disable-next=too-many-positional-arguments
    def generate_batches(
        self,
        batch_size: int,
        shuffle: bool = False,
        drop_remainder: bool = True,
        replacement: bool = False,
        poisson: bool = False,
    ) -> Iterator[Batch]:
        # inherited signature; pylint: disable=too-many-arguments
        if poisson:
            generator = self._generate_batches_poisson(
                batch_size=batch_size,
                drop_remainder=drop_remainder,
            )
        else:
            generator = self._generate_batches_batching(
                batch_size=batch_size,
                drop_remainder=drop_remainder,
                shuffle=shuffle,
                replacement=replacement,
            )
        yield from generator

    def _generate_batches_batching(
        self,
        batch_size: int,
        drop_remainder: bool,
        shuffle: bool,
        replacement: bool,
    ) -> Iterator[Batch]:
        """Backend to `generate_batches` when `poisson=False`."""
        # Start setting up the dataset, then setup optional shuffling.
        dataset, _, n_batches, none_pads = self._prepare_dataset(
            batch_size=batch_size, drop_remainder=drop_remainder
        )
        if shuffle:
            dataset = self._setup_shuffling(
                dataset, batch_size=batch_size, replacement=replacement
            )
        # Set up batching, opt. in padded or ragged mode.
        batch = get_batch_function(self.batch_mode, dataset=dataset)
        dataset = batch(batch_size=batch_size, drop_remainder=drop_remainder)
        dataset = dataset.take(n_batches).map(lambda *s: (*s, *none_pads))
        yield from dataset

    def _generate_batches_poisson(
        self,
        batch_size: int,
        drop_remainder: bool,
    ) -> Iterator[Batch]:
        """Backend to `generate_batches` when `poisson=True`."""
        # Start setting up the dataset, then setup shuffling with replacement.
        dataset, n_samples, n_batches, none_pads = self._prepare_dataset(
            batch_size=batch_size, drop_remainder=drop_remainder
        )
        dataset = self._setup_shuffling(
            dataset, batch_size=batch_size, replacement=True
        )
        # Draw batches' size, that follows a Binomial law.
        srate = batch_size / n_samples
        sizes = self.rng.binomial(n=n_samples, p=srate, size=n_batches)
        # Fetch and batch up samples manually.
        itersamples = iter(dataset)
        stack = get_stack_function(self.batch_mode)
        for size in sizes:
            if not size:  # skip empty batches (edge case)
                continue
            samples = [
                # infinite iterator; pylint: disable=stop-iteration-return
                next(itersamples)
                for _ in range(size)
            ]
            batch = tf.nest.map_structure(stack, *samples)
            yield (*batch, *none_pads)  # type: ignore

    def _prepare_dataset(
        self,
        batch_size: int,
        drop_remainder: bool,
    ) -> Tuple[tf.data.Dataset, int, int, List[None]]:
        """Run initial preparations to generate batches from the dataset.

        Return
        - the wrapped dataset (optionally with some transforms),
        - the number of samples in the initial dataset
        - the number of batches that should be yielded
        - a list of None values to end up adding to complete batches
        """
        dataset = self.dataset
        if self._dspecs.single_el:
            dataset = dataset.map(lambda x: (x,))
        none_pads = [None] * self._dspecs.n_padding
        # Compute the number of batches that are to be yielded.
        n_samples = self._dspecs.n_samples
        n_batches = n_samples // batch_size
        n_batches += (not drop_remainder) and (n_samples % batch_size)
        # Return that information.
        return dataset, n_samples, n_batches, none_pads

    def _setup_shuffling(
        self,
        dataset: tf.data.Dataset,
        batch_size: int,
        replacement: bool,
    ) -> tf.data.Dataset:
        """Transform a dataset into a shuffled one, opt. with replacement.

        Use `self.buffer_size` or `10 * batch_size` as buffer size.
        If `replacement`, the returned dataset is an infinite iterator.
        """
        if replacement:
            dataset = dataset.repeat(count=None)
        return dataset.shuffle(
            seed=int(self.rng.integers(2**63)),
            buffer_size=self.buffer_size or batch_size * 10,
        )


# All following are backend tools for 'TensorflowDataset'.


@dataclasses.dataclass
class TensorflowDatasetSpec:
    """Internal dataclass to specify information about a TF Dataset to wrap."""

    single_el: bool = False
    n_padding: int = 0
    n_samples: int = 0
    data_type: str = "float32"
    input_shp: List[Optional[int]] = dataclasses.field(default_factory=list)
    y_classes: Optional[Set[int]] = None


def parse_and_validate_tensorflow_dataset(
    dataset: tf.data.Dataset,
) -> TensorflowDatasetSpec:
    """Analyze and validate a tensorflow Dataset that needs wrapping.

    Return a `TensorflowDatasetSpec` instance that gathers collected
    metadata, for Dataset post-processing and DataSpecs reporting.
    """
    info = TensorflowDatasetSpec()
    # Type-check the dataset and its specs' overall shape.
    if not isinstance(dataset, tf.data.Dataset):
        raise TypeError("'dataset' should be a 'tf.data.Dataset'.")
    spec = dataset.element_spec
    if not isinstance(spec, tuple):
        spec = (spec,)
        info.single_el = True
    if not 1 <= len(spec) <= 3:
        raise ValueError(
            "Input tensorflow.data.Dataset should yield batches of "
            f"1 to 3 elements (not {len(spec)}), denoting, in that "
            "order: input features, target labels and sample weights."
        )
    # Gather cardinality and need for y and/or w elements' filling.
    info.n_padding = 3 - len(spec)
    if int(tf.version.VERSION.split(".", 2)[1]) >= 13:
        info.n_samples = int(dataset.cardinality().numpy())  # type: ignore
        if info.n_samples == tf.data.UNKNOWN_CARDINALITY:
            info.n_samples = 0  # force evaluating via next logic branch
    if not info.n_samples:
        info.n_samples = sum(1 for _ in dataset.scan(0, lambda s, _: (s, s)))
    # Gather information about input features.
    inputs = tf.nest.flatten(spec[0])[0]
    info.data_type = inputs.dtype.name
    info.input_shp = list(inputs.shape)
    # Gather information about target labels (if any).
    if len(spec) >= 2:
        labels = tf.nest.flatten(spec[1])[0]
        if isinstance(labels, tf.TensorSpec) and labels.dtype.is_integer:
            unique = dataset.map(lambda *x: tf.nest.flatten(x[1])[0]).unique()
            info.y_classes = set(tf.stack(list(unique)).numpy())
    # Return the gathered information.
    return info


def warn_if_dataset_is_likely_batched(
    dataset: tf.data.Dataset,
) -> None:
    """Warn users about possibly-batched input tensorflow Dataset objects."""
    # Gather the shape of all non-None elements in a dataset sample.
    shapes = [
        spec.shape
        for spec in tf.nest.flatten(dataset.element_spec)
        if isinstance(spec, tf.TensorSpec)
    ]
    # Warn if all shapes have a defined and identical first dimension.
    if (
        all(shape.ndims > 0 for shape in shapes)
        and len(shapes) > 1
        and all(shape[0] is not None for shape in shapes)
        and all(shape[0] == shapes[0][0] for shape in shapes[1:])
    ):
        warnings.warn(
            "The input 'tf.data.Dataset' wraps tensors that all share the "
            "same first dimension; this may be the signed of batched data, "
            "that will cause issues. Please verify that the input dataset "
            "is properly yielding unbatched samples. If so, you may ignore "
            "this warning.",
            category=RuntimeWarning,
            stacklevel=2,
        )


def get_batch_function(
    batch_mode: BatchMode,
    dataset: tf.data.Dataset,
) -> Callable[..., tf.data.Dataset]:
    """Return a bound method to batch an input `tf.data.Dataset`."""
    if batch_mode == "default":
        return dataset.batch
    if batch_mode == "padded":
        return dataset.padded_batch
    if batch_mode == "ragged":
        return dataset.ragged_batch
    raise TypeError(
        "Invalid value for 'batch_mode': should be one of "
        f"{{'default', 'padded', 'ragged'}}, not '{batch_mode}'."
    )


def get_stack_function(
    batch_mode: BatchMode,
) -> Callable[[Union[List[None], List[tf.Tensor]]], Optional[tf.Tensor]]:
    """Return a function to stack sample-wise atomic elements."""
    if batch_mode == "default":
        return _stack_default  # type: ignore
    if batch_mode == "padded":
        return _stack_padded  # type: ignore
    if batch_mode == "ragged":
        return _stack_ragged  # type: ignore
    raise TypeError(
        "Invalid value for 'batch_mode': should be one of "
        f"{{'default', 'padded', 'ragged'}}, not '{batch_mode}'."
    )


def _stack_default(
    *samples: Union[None, tf.Tensor],
) -> Optional[tf.Tensor]:
    """Stack sample-wise atomic elements."""
    if samples[0] is None:
        return None
    return tf.stack(samples)


def _stack_padded(
    *samples: Union[None, tf.Tensor],
) -> Optional[tf.Tensor]:
    """Pad and stack sample-wise atomic elements."""
    if samples[0] is None:
        return None
    return tf.ragged.stack(samples).to_tensor()


def _stack_ragged(
    *samples: Union[None, tf.Tensor],
) -> Optional[tf.Tensor]:
    """Stack sample-wise atomic elements into a tf.RaggedTensor."""
    if samples[0] is None:
        return None
    return tf.ragged.stack(samples)
