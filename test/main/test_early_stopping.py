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

"""Unit tests for 'declearn.main.utils.EarlyStopping'."""

from declearn.main.utils import EarlyStopping


class TestEarlyStopping:
    """Unit tests for 'declearn.main.utils.EarlyStopping'."""

    def test_keep_training_initial(self) -> None:
        """Test that a brand new EarlyStopping indicates to train."""
        early_stop = EarlyStopping()
        assert early_stop.keep_training

    def test_update_first(self) -> None:
        """Test that an instantiated EarlyStopping's update works."""
        early_stop = EarlyStopping()
        keep_going = early_stop.update(1.0)
        assert keep_going
        assert keep_going == early_stop.keep_training

    def test_update_twice(self) -> None:
        """Test that an EarlyStopping can be reached in a simple case."""
        early_stop = EarlyStopping(tolerance=0.0, patience=1, decrease=True)
        assert early_stop.update(1.0)
        assert not early_stop.update(1.0)
        assert not early_stop.keep_training

    def test_reset_after_stopping(self) -> None:
        """Test that 'EarlyStopping.reset()' works properly."""
        # Reach the criterion once.
        early_stop = EarlyStopping(tolerance=0.0, patience=1, decrease=True)
        assert early_stop.update(1.0)
        assert not early_stop.update(1.0)
        assert not early_stop.keep_training
        # Reset and test that the criterion has been properly reset.
        early_stop.reset()
        assert early_stop.keep_training
        assert early_stop.update(1.0)
        # Reach the criterion for the second time.
        assert not early_stop.update(1.0)
        assert not early_stop.keep_training

    def test_with_two_steps_patience(self) -> None:
        """Test an EarlyStopping criterion with 2-steps patience."""
        early_stop = EarlyStopping(tolerance=0.0, patience=2, decrease=True)
        assert early_stop.update(1.0)
        assert early_stop.update(1.5)  # patience tempers stopping
        assert early_stop.update(0.0)  # patience is reset
        assert early_stop.update(0.5)  # patience tempers stopping
        assert not early_stop.update(0.2)  # patience is exhausted

    def test_with_absolute_tolerance_positive(self) -> None:
        """Test an EarlyStopping criterion with 0.2 absolute tolerance."""
        early_stop = EarlyStopping(tolerance=0.2, patience=1, decrease=True)
        assert early_stop.update(1.0)
        assert early_stop.update(0.7)
        assert not early_stop.update(0.6)  # progress below tolerance

    def test_with_absolute_tolerance_negative(self) -> None:
        """Test an EarlyStopping criterion with -0.5 absolute tolerance."""
        early_stop = EarlyStopping(tolerance=-0.5, patience=1, decrease=True)
        assert early_stop.update(1.0)
        assert early_stop.update(1.2)  # regression below tolerance
        assert not early_stop.update(1.6)  # regression above tolerance

    def test_with_relative_tolerance_positive(self) -> None:
        """Test an EarlyStopping criterion with 0.1 relative tolerance."""
        early_stop = EarlyStopping(
            tolerance=0.1, patience=1, decrease=True, relative=True
        )
        assert early_stop.update(1.0)
        assert early_stop.update(0.8)  # progress above tolerance
        assert not early_stop.update(0.75)  # progress below tolerance

    def test_with_relative_tolerance_negative(self) -> None:
        """Test an EarlyStopping criterion with -0.1 relative tolerance."""
        early_stop = EarlyStopping(
            tolerance=-0.1, patience=1, decrease=True, relative=True
        )
        assert early_stop.update(1.0)
        assert early_stop.update(0.80)  # progress
        assert early_stop.update(0.85)  # regression below tolerance
        assert not early_stop.update(0.89)  # regression above tolerance

    def test_with_increasing_metric(self) -> None:
        """Test an EarlyStopping that monitors an increasing metric."""
        early_stop = EarlyStopping(tolerance=0.0, patience=1, decrease=False)
        assert early_stop.update(1.0)
        assert early_stop.update(2.0)  # progress
        assert not early_stop.update(1.5)  # regression (no patience/tolerance)
