import json
from pathlib import Path

from firstcoder.ci import parse_pytest_output


FIXTURES = Path(__file__).parent / "fixtures" / "pytest_logs"


def _fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def test_parses_assertion_expected_actual_and_posix_location() -> None:
    report = parse_pytest_output(_fixture("assertion.txt"), command="python -m pytest", exit_code=1)

    assert report.status == "failed"
    assert report.failed_count == 1
    failure = report.failures[0]
    assert failure.node_id == "tests/test_math.py::test_invoice_total"
    assert failure.phase == "call"
    assert failure.failure_kind == "assertion"
    assert failure.exception_type == "AssertionError"
    assert failure.expected == "12"
    assert failure.actual == "10"
    assert failure.source_locations[0].path == "src/invoice.py"
    assert failure.source_locations[0].line == 14
    assert len(failure.fingerprint) == 24


def test_parses_multiple_parameterized_failures_and_exception_types() -> None:
    report = parse_pytest_output(_fixture("multi_failure.txt"), exit_code=1)

    assert report.failed_count == 2
    assert [failure.node_id for failure in report.failures] == [
        "tests/test_parser.py::test_parse_valid[user]",
        "tests/test_parser.py::test_parse_valid[admin]",
    ]
    assert [failure.exception_type for failure in report.failures] == ["ValueError", "TypeError"]
    assert [failure.source_locations[0].line for failure in report.failures] == [20, 31]


def test_parses_setup_and_teardown_errors() -> None:
    report = parse_pytest_output(_fixture("setup_teardown.txt"), exit_code=1)

    assert report.error_count == 2
    assert [failure.phase for failure in report.failures] == ["setup", "teardown"]
    assert [failure.exception_type for failure in report.failures] == ["RuntimeError", "ValueError"]


def test_parses_collection_import_error() -> None:
    report = parse_pytest_output(_fixture("collection_import.txt"), exit_code=2)

    assert report.status == "error"
    assert report.error_count == 1
    failure = report.failures[0]
    assert failure.node_id == "tests/test_service.py"
    assert failure.phase == "collection"
    assert failure.failure_kind == "import_error"
    assert failure.exception_type == "ImportError"


def test_parses_windows_source_path_and_explicit_expected_actual() -> None:
    report = parse_pytest_output(_fixture("windows.txt"), exit_code=1)

    failure = report.failures[0]
    assert failure.source_locations[0].path == "C:/work/repo/src/config.py"
    assert failure.expected == "enabled"
    assert failure.actual == "disabled"


def test_cleans_ansi_and_fingerprint_ignores_temp_prefix_address_time_and_duration() -> None:
    first = """\
\x1b[31mFAILED\x1b[0m tests/test_x.py::test_x - ValueError: bad object 0xABCDEF at 2026-07-14T10:11:12Z
/private/tmp/run-a/src/x.py:9: ValueError
1 failed in 0.10s
"""
    second = """\
FAILED tests/test_x.py::test_x - ValueError: bad object 0x123456 at 2027-01-01T00:00:00Z
/private/tmp/run-b/src/x.py:9: ValueError
1 failed in 9.99s
"""

    one = parse_pytest_output(first, exit_code=1).failures[0]
    two = parse_pytest_output(second, exit_code=1).failures[0]

    assert one.fingerprint == two.fingerprint
    assert "\x1b" not in one.message


def test_fingerprint_changes_for_node_or_exception() -> None:
    base = "FAILED tests/test_x.py::test_x - ValueError: bad value\n1 failed in 0.1s"
    changed_node = "FAILED tests/test_x.py::test_y - ValueError: bad value\n1 failed in 0.1s"
    changed_exception = "FAILED tests/test_x.py::test_x - TypeError: bad value\n1 failed in 0.1s"

    fingerprints = {
        parse_pytest_output(text, exit_code=1).failures[0].fingerprint
        for text in (base, changed_node, changed_exception)
    }

    assert len(fingerprints) == 3


def test_truncates_bounded_raw_output_without_losing_parsed_failure() -> None:
    output = _fixture("assertion.txt") + ("noise\n" * 100)

    report = parse_pytest_output(output, exit_code=1, max_raw_chars=180)

    assert report.truncated is True
    assert len(report.bounded_raw_output) <= 180
    assert report.failures[0].node_id == "tests/test_math.py::test_invoice_total"


def test_malformed_and_no_tests_outputs_degrade_without_exception() -> None:
    malformed = parse_pytest_output("\x00not really pytest { broken", exit_code=3)
    no_tests = parse_pytest_output("no tests ran in 0.01s", exit_code=5)

    assert malformed.status == "unknown"
    assert malformed.failures == []
    assert no_tests.status == "no_tests"
    assert no_tests.failures == []


def test_report_is_json_serializable_and_preserves_command_duration() -> None:
    report = parse_pytest_output(
        "1 passed, 2 skipped in 0.25s",
        command="python -m pytest -q",
        exit_code=0,
    )

    payload = report.to_dict()
    assert payload["command"] == "python -m pytest -q"
    assert payload["status"] == "passed"
    assert payload["passed_count"] == 1
    assert payload["skipped_count"] == 2
    assert payload["duration"] == 0.25
    json.dumps(payload)
