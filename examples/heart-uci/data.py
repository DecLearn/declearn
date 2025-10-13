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

"""Script to download and pre-process the UCI Heart Disease Dataset."""

import argparse
import os

from declearn.dataset.examples import load_heart_uci


DATADIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
NAMES = ("cleveland", "hungarian", "switzerland", "va")


# Code executed when the script is called directly.
if __name__ == "__main__":
    # Parse commandline parameters.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder",
        type=str,
        default=DATADIR,
        help="folder where to write output csv files",
    )
    parser.add_argument(
        "names",
        action="append",
        nargs="+",
        help="name(s) of client center(s), data from which to prepare",
        choices=["cleveland", "hungarian", "switzerland", "va"],
    )
    args = parser.parse_args()
    # Download and pre-process the selected dataset(s).
    for name in args.names:
        load_heart_uci(name=name, folder=args.folder)
