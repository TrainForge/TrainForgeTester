from __future__ import annotations

from trainforge.standard_checks import standard_check_ids, standard_check_questions


def test_standard_check_accessors_have_expected_length() -> None:
    assert len(standard_check_ids()) == 20
    assert len(standard_check_questions()) == 20

