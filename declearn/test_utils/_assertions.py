# coding: utf-8

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
