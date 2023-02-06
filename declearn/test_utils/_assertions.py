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

"""Custom "assert" functions commonly used in declearn tests."""

import json
from typing import Any, Dict


from declearn.utils import json_pack, json_unpack


__all__ = [
    "assert_json_serializable_dict",
]


def assert_json_serializable_dict(sdict: Dict[str, Any]) -> None:
    """Assert that an input is JSON-serializable using declearn hooks.

    This function tries to dump the input dict into a JSON string,
    then to reload it. It does so using `declearn.utils.json_pack`
    and `json_unpack` functions to extend JSON (en|de)coding. It
    also asserts that the recovered dict is similar to the initial
    one.

    Parameters
    ----------
    sdict: Dict[str, Any]
        Dictionary, the JSON-serializability of which to assert.

    Raises
    ------
    AssertionError:
        If `sdict` or the JSON-reloaded object is not a dict, or
        if the latter has different keys and/or values compared
        to the former.
    Exception:
        Other exceptions may be raised if the JSON encoding (or
        decoding) operation goes wrong.
    """
    assert isinstance(sdict, dict)
    dump = json.dumps(sdict, default=json_pack)
    load = json.loads(dump, object_hook=json_unpack)
    assert isinstance(load, dict)
    assert load.keys() == sdict.keys()
    assert all(load[key] == sdict[key] for key in sdict)
