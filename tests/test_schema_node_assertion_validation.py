"""NodeAssertion schema validation — args_match + must_fire combo rules.

The args_match filter is only meaningful when must_fire=True (we expect
the node to fire AND want to check args). The negative-assertion case
("node X must not fire with these args") is ambiguous and rejected at
validation time so users get a clear error instead of silent foot-gun.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from trainforge.schema import NodeAssertion


def test_must_fire_true_with_args_match_is_valid() -> None:
    na = NodeAssertion(
        node_name="refund_handler",
        must_fire=True,
        args_match={"invoice_id": "inv-1"},
    )
    assert na.must_fire is True
    assert na.args_match == {"invoice_id": "inv-1"}


def test_must_fire_false_without_args_match_is_valid() -> None:
    """The negative assertion 'node must not fire' is fine on its own."""
    na = NodeAssertion(node_name="danger_node", must_fire=False)
    assert na.must_fire is False
    assert na.args_match is None


def test_must_fire_false_with_args_match_is_rejected() -> None:
    """The ambiguous combo must raise at validation time."""
    with pytest.raises(ValidationError) as exc:
        NodeAssertion(
            node_name="danger_node",
            must_fire=False,
            args_match={"amount": 1000},
        )
    assert "args_match is only valid when must_fire=True" in str(exc.value)


def test_default_must_fire_true_with_args_match_is_valid() -> None:
    """must_fire defaults to True; args_match should be accepted with default."""
    na = NodeAssertion(node_name="node_x", args_match={"k": "v"})
    assert na.must_fire is True
    assert na.args_match == {"k": "v"}


def test_no_args_match_no_must_fire_explicit_defaults() -> None:
    na = NodeAssertion(node_name="node_y")
    assert na.must_fire is True
    assert na.args_match is None
