"""Resolve ``--agent module:callable`` strings into a Python callable.

Mirrors uvicorn / gunicorn syntax:

- ``my_pkg.my_module:run_agent`` imports ``my_pkg.my_module`` and grabs
  ``run_agent`` (which must be a callable).
- ``my_pkg.my_module:make_agent()`` imports the module, grabs
  ``make_agent``, calls it with no arguments, and uses the returned value
  as the agent callable. Use this for factories that need to set up
  resources (clients, configs, models) before the agent is ready to run.

The returned value is suitable to hand to
:class:`trainforge.transport.InProcessTransport`.
"""
from __future__ import annotations

import importlib
import re
from typing import Any

_SPEC_RE = re.compile(
    r"""
    ^                                       # start
    (?P<module>[\w][\w\.]*)                 # dotted module path
    :                                       # separator
    (?P<attr>[\w][\w\.]*)                   # attribute (callable or factory)
    (?P<call>\(\))?                         # optional factory invocation
    $                                       # end
    """,
    re.VERBOSE,
)


class AgentResolutionError(Exception):
    """The ``--agent`` spec could not be resolved into a callable."""


def resolve_in_process_agent(spec: str) -> Any:
    """Resolve a uvicorn-style ``module:callable`` string.

    Examples:

        resolve_in_process_agent("examples.quickstart.agent:run")
        resolve_in_process_agent("examples.quickstart.agent:make_agent()")

    Raises :class:`AgentResolutionError` on any failure (import, attribute,
    factory call, or non-callable result).
    """
    match = _SPEC_RE.match(spec.strip())
    if not match:
        raise AgentResolutionError(
            f"invalid --agent spec {spec!r}; expected 'module:callable' or 'module:factory()'"
        )

    module_name = match.group("module")
    attr_path = match.group("attr")
    is_factory = match.group("call") is not None

    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise AgentResolutionError(
            f"could not import module {module_name!r} from --agent {spec!r}: {exc}"
        ) from exc

    target: Any = module
    for part in attr_path.split("."):
        try:
            target = getattr(target, part)
        except AttributeError as exc:
            raise AgentResolutionError(
                f"module {module_name!r} has no attribute path {attr_path!r}: {exc}"
            ) from exc

    if is_factory:
        if not callable(target):
            raise AgentResolutionError(
                f"factory {spec!r} is not callable: got {type(target).__name__}"
            )
        try:
            target = target()
        except Exception as exc:  # noqa: BLE001
            raise AgentResolutionError(
                f"factory {spec!r} raised {type(exc).__name__}: {exc}"
            ) from exc

    if not callable(target):
        raise AgentResolutionError(
            f"--agent {spec!r} did not resolve to a callable (got {type(target).__name__})"
        )

    return target
