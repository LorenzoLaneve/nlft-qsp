import json
from pathlib import Path
from typing import Type, TypeVar, TextIO

import numpy as np

from .numerics import complex_type, float_type

T = TypeVar("T")

TYPE_REGISTRY: dict = {}

COMPLEX_TYPE_TAG = "@qspx/complex"

NDARRAY_COMPLEX_TYPE_TAG = "@qspx/ndarray/complex"
NDARRAY_FLOAT_TYPE_TAG = "@qspx/ndarray/float"


def _parse_json(content):
    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}") from e

    if not isinstance(data, dict):
        raise ValueError(
            f"Expected root JSON payload to be an object/dict with a '@type' field, "
            f"but got {type(data).__name__}."
        )

    if not data.get("@type"):
        raise ValueError("Missing required '@type' field in JSON payload.")

    return data

def _value_to_json(val):
    if isinstance(val, complex):
        return {
            "@type": COMPLEX_TYPE_TAG,
            "re": np.real(val),
            "im": np.imag(val),
        }
    
    if isinstance(val, np.ndarray):
        if np.iscomplexobj(val):
            return {
                "@type": NDARRAY_COMPLEX_TYPE_TAG,
                "shape": list(val.shape),
                "values": _value_to_json(val.tolist()),
            }
        
        return {
            "@type": NDARRAY_FLOAT_TYPE_TAG,
            "shape": list(val.shape),
            "values": _value_to_json(val.tolist()),
        }

    if isinstance(val, list):
        return [_value_to_json(item) for item in val]
    if isinstance(val, dict):
        return {k: _value_to_json(v) for k, v in val.items()}
    return val

def _json_to_value(json):
    if isinstance(json, dict):
        if json.get("@type") == COMPLEX_TYPE_TAG:
            return complex_type(json["re"], json["im"])
        elif json.get("@type") == NDARRAY_COMPLEX_TYPE_TAG:
            return np.array(_json_to_value(json["values"]), dtype=complex_type)
        elif json.get("@type") == NDARRAY_FLOAT_TYPE_TAG:
            return np.array(_json_to_value(json["values"]), dtype=float_type)

    if isinstance(json, list):
        return [_json_to_value(item) for item in json]
    
    return json


def serializable(type_tag: str, fields: dict):
    """Class decorator to handle JSON serialization.

    Args:
        type_tag: The identifier tag added/expected in JSON as `@type`.
        fields: A mapping of { python_attribute_name: json_field_name }.
    """

    def decorator(cls: Type[T]) -> Type[T]:
        json_to_attr = {v: k for k, v in fields.items()}

        def _dump_json_str(self, **json_kwargs) -> str:
            """Serialize instance to a JSON string."""
            payload: dict = {"@type": type_tag}

            for attr_name, json_key in fields.items():
                if hasattr(self, attr_name):
                    payload[json_key] = _value_to_json(getattr(self, attr_name))

            return json.dumps(payload, indent=2)

        def dump_json(self, file_or_path: str | Path | TextIO, **json_kwargs):
            """Serialize instance to a file path or open file stream."""
            json_str = self._dump_json_str(**json_kwargs)

            if isinstance(file_or_path, (str, Path)):
                Path(file_or_path).write_text(json_str, encoding="utf-8")
            else:
                file_or_path.write(json_str)

        @classmethod
        def _load_json_data(cls: Type[T], json_data: dict) -> T:
            """Deserialize instance from a JSON dictionary."""
            
            file_type = json_data.get("@type")
            if file_type != type_tag:
                raise ValueError(
                    f"Type mismatch: Expected @type '{type_tag}', but got '{file_type}'."
                )

            init_kwargs = {}
            for json_key, attr_name in json_to_attr.items():
                if json_key in json_data:
                    init_kwargs[attr_name] = _json_to_value(json_data[json_key])

            return cls(**init_kwargs)

        @classmethod
        def load_json(cls: Type[T], file_or_path: str | Path | TextIO) -> T:
            """Deserialize instance from a file path or open file stream."""
            if isinstance(file_or_path, (str, Path)):
                content = Path(file_or_path).read_text(encoding="utf-8")
            else:
                content = file_or_path.read()

            return cls._load_json_data(_parse_json(content))

        cls.dump_json = dump_json
        cls.load_json = load_json
        cls._dump_json_str = _dump_json_str
        cls._load_json_data = _load_json_data

        cls._type_tag = type_tag
        cls._field_mapping = fields
        TYPE_REGISTRY[type_tag] = cls

        return cls

    return decorator


def load_any(file_or_path: str | Path | TextIO) -> object:
    """Load any registered object dynamically based on its root '@type'.
    
    Args:
        file_or_path: File path (str or Path) or an open text stream.
        
    Returns:
        An instantiated instance of the registered class matching `@type`.
        
    Raises:
        ValueError: If the JSON file contains invalid JSON.
        TypeError: If `@type` is missing or unknown.
    """
    if isinstance(file_or_path, (str, Path)):
        content = Path(file_or_path).read_text(encoding="utf-8")
    else:
        content = file_or_path.read()

    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}") from e

    if not isinstance(data, dict):
        raise ValueError(
            f"Expected root JSON payload to be an object/dict with a '@type' field, "
            f"but got {type(data).__name__}."
        )

    data = _parse_json(content)

    type_tag = data.get("@type")
    if type_tag not in TYPE_REGISTRY:
        registered_tags = list(TYPE_REGISTRY.keys())
        raise TypeError(
            f"Unknown '@type': '{type_tag}'. Registered types are: {registered_tags}"
        )

    return TYPE_REGISTRY[type_tag]._load_json_data(data)