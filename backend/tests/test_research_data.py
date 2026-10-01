from decimal import Decimal

from xcrs.ingest.research_data import legacy_parent_id, legacy_role_id, parse_price


def test_legacy_role_id_decodes_digit_encoded_ids():
    assert legacy_role_id(100) == 1  # role 1 topic
    assert legacy_role_id(10000) == 1  # role 1 concept
    assert legacy_role_id(60203) == 6
    assert legacy_role_id(1000) == 10  # role 10 topic
    assert legacy_role_id(100001) == 10


def test_legacy_parent_id():
    assert legacy_parent_id(602) is None  # top-level node of role 6
    assert legacy_parent_id(60203) == 602
    assert legacy_parent_id(1000) is None  # top-level node of role 10
    assert legacy_parent_id(100001) == 1000


def test_parse_price():
    assert parse_price("Free") == Decimal(0)
    assert parse_price("₺299.99") == Decimal("299.99")
