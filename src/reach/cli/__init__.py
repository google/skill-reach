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

"""Provide command-line interface entry point and command registration for Reach."""

from __future__ import annotations

import sys
from functools import cache
from typing import TYPE_CHECKING

from cyclopts.exceptions import CycloptsError
from pydantic import ValidationError as PydanticValidationError

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from cyclopts import App

from reach.views import build_console, error_panel, help_console

from . import (  # noqa: F401
    check,
    clean,
    cluster,
    completion,
    diff,
    doctor,
    eval,
    init,
    lint,
    optimize,
    overlap,
    query,
    sweep,
    view,
)
from .app import _explain, _verbs, app
from .eval import EVAL_REQUIRED
from .flags import (
    AgentName,
    Format,
    StudyFlags,
    Vary,
    build_config,
    parse_agent_options,
)

__all__ = [
    "EVAL_REQUIRED",
    "AgentName",
    "Format",
    "StudyFlags",
    "Vary",
    "_verbs",
    "app",
    "build_config",
    "main",
    "parse_agent_options",
]


@cache
def _value_options() -> frozenset[str]:
    """Return option names that consume at least one value token in any command."""
    names: set[str] = set()
    for command in _commands(app):
        for argument in command.assemble_argument_collection():
            if argument.token_count()[0] < 1:
                continue
            names.update(name for name in argument.names if name.startswith("-"))
    return frozenset(names)


def _commands(root: App) -> Iterator[App]:
    """Yield an application and every subcommand registered beneath it."""
    yield root
    for name in root:
        if not name.startswith("-"):
            yield from _commands(root[name])


def _verb_index(raw: Sequence[str], verbs: set[str], value_options: frozenset[str]) -> int | None:
    """Locate the subcommand argument, skipping values consumed by preceding options."""
    skip_value = False
    for i, arg in enumerate(raw):
        if arg == "--":
            return None
        if skip_value:
            skip_value = False
        elif arg.startswith("-"):
            skip_value = "=" not in arg and arg in value_options
        elif arg in verbs:
            return i
    return None


def _reorder_argv(argv: list[str] | None) -> list[str]:
    """Normalize argv by positioning recognized subcommands before leading options."""
    raw = sys.argv[1:] if argv is None else list(argv)
    if not raw or raw[0] == "__complete":
        return raw

    verbs = set(_verbs())
    verb_idx = _verb_index(raw, verbs, frozenset())
    if verb_idx is None or verb_idx == 0:
        return raw

    # A verb preceded by an option may in fact be that option's value, such as
    # `reach --skill diff lint`, so resolve the ambiguity using declared option arity.
    if raw[verb_idx - 1].startswith("-"):
        precise = _verb_index(raw, verbs, _value_options())
        verb_idx = verb_idx if precise is None else precise
        if verb_idx == 0:
            return raw

    before = raw[:verb_idx]
    verb = raw[verb_idx]
    after = raw[verb_idx + 1 :]
    return [verb, *after, *before]


def main(argv: list[str] | None = None) -> int:
    """Execute the CLI application and translate exceptions into exit codes."""
    effective_argv = _reorder_argv(argv)
    try:
        outcome = app(
            effective_argv,
            console=help_console(),
            error_console=build_console(),
            exit_on_error=False,
            error_formatter=lambda error: error_panel(_explain(error)),
        )
    except BrokenPipeError:
        return 0
    except CycloptsError:
        return 2
    except (
        ValueError,
        KeyError,
        OSError,
        RuntimeError,
        PydanticValidationError,
    ) as error:
        said = error.args[0] if isinstance(error, KeyError) else str(error)
        build_console().print(error_panel([str(said)]))
        return 2
    return 0 if outcome is None else outcome


if __name__ == "__main__":
    raise SystemExit(main())
