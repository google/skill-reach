# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Private filesystem I/O helpers for model serialization."""

from __future__ import annotations

import os
import uuid
from pathlib import Path

from pydantic import BaseModel

__all__ = [
    "atomic_write_text",
    "read_model",
    "write_model",
]


def atomic_write_text(
    path: Path | str,
    content: str,
    encoding: str = "utf-8",
) -> Path:
    """Atomically write text to disk using a process- and thread-unique temporary file."""
    resolved = Path(path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temp_path = resolved.with_suffix(f".tmp.{os.getpid()}_{uuid.uuid4().hex}")
    temp_path.write_text(content, encoding=encoding)
    temp_path.replace(resolved)
    return resolved


def read_model[M: BaseModel](model_cls: type[M], path: Path | str) -> M:
    """Read and validate a Pydantic model from disk."""
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        msg = f"Model source file not found or is not a regular file: {resolved}"
        raise FileNotFoundError(msg)
    return model_cls.model_validate_json(resolved.read_bytes())


def write_model(model: BaseModel, path: Path | str) -> Path:
    """Serialize a Pydantic model to disk as formatted JSON atomically."""
    return atomic_write_text(path, model.model_dump_json(indent=2) + "\n", encoding="utf-8")
