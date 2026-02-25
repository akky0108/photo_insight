from __future__ import annotations

import pytest


def test_validate_input_contract_ok(required_columns):
    from photo_insight.batch_processor.evaluation_rank.contract import validate_input_contract

    validate_input_contract(header=required_columns, csv_path=None)


def test_validate_input_contract_missing_raises(required_columns):
    from photo_insight.batch_processor.evaluation_rank.contract import validate_input_contract

    header = [c for c in required_columns if c != "file_name"]
    with pytest.raises(Exception):
        validate_input_contract(header=header, csv_path=None)
