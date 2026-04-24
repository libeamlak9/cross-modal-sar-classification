from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


DEFAULT_TEMPLATES_OPTICAL = [
    "an optical satellite image of {}.",
    "a remote sensing image of {}.",
    "a satellite view of {}.",
]

DEFAULT_TEMPLATES_SAR = [
    "a SAR image of {}.",
    "a synthetic aperture radar image of {}.",
    "a radar image of {}.",
]

DEFAULT_TEMPLATES = DEFAULT_TEMPLATES_OPTICAL + DEFAULT_TEMPLATES_SAR

_MAPPING_CACHE: Dict[Path, Dict[str, str]] = {}
_LOGGED_ROOTS: set[Path] = set()


def _fallback_classname(class_id: str) -> str:
    return f"class {class_id}" if class_id.isdigit() else class_id


def _mapping_path(dataset_root: Path) -> Path:
    return dataset_root / "classnames.json"


def _validate_mapping(mapping: Dict[str, str]) -> Dict[str, str]:
    expected = [str(i) for i in range(17)]
    missing = [cid for cid in expected if cid not in mapping]
    if missing:
        raise AssertionError(f"classnames.json missing ids: {missing}")
    return {str(k): str(v) for k, v in mapping.items()}


def _load_lcz_classnames(dataset_root: Path) -> Dict[str, str]:
    mapping_path = _mapping_path(dataset_root)
    if mapping_path in _MAPPING_CACHE:
        return _MAPPING_CACHE[mapping_path]
    if not mapping_path.exists():
        _MAPPING_CACHE[mapping_path] = {}
        return _MAPPING_CACHE[mapping_path]
    with mapping_path.open("r", encoding="utf-8") as f:
        mapping = json.load(f)
    if not isinstance(mapping, dict):
        raise ValueError("classnames.json must contain a JSON object mapping IDs to names.")
    mapping = _validate_mapping(mapping)
    _MAPPING_CACHE[mapping_path] = mapping
    if dataset_root not in _LOGGED_ROOTS:
        sample_id = "0"
        sample_name = mapping.get(sample_id, _fallback_classname(sample_id))
        example_prompt = DEFAULT_TEMPLATES_SAR[0].format(sample_name)
        print(
            f"Loaded {len(mapping)} classnames from {mapping_path} | "
            f"Example: id {sample_id} -> \"{sample_name}\" | "
            f"Example prompt: \"{example_prompt}\""
        )
        _LOGGED_ROOTS.add(dataset_root)
    return mapping


def resolve_classname(dataset_root: Path, class_id: str | int) -> str:
    mapping = _load_lcz_classnames(dataset_root)
    cid = str(class_id)
    if mapping:
        if cid not in mapping:
            raise AssertionError(f"Missing class id in classnames.json: {cid}")
        return mapping[cid]
    return _fallback_classname(cid)


def get_classnames(dataset_root: Path, class_ids: Iterable[str | int]) -> List[str]:
    mapping = _load_lcz_classnames(dataset_root)
    classnames: List[str] = []
    for cid in class_ids:
        cid_str = str(cid)
        if mapping:
            if cid_str not in mapping:
                raise AssertionError(f"Missing class id in classnames.json: {cid_str}")
            classnames.append(mapping[cid_str])
        else:
            classnames.append(_fallback_classname(cid_str))
    return classnames

