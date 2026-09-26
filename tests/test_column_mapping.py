import pytest

from openece.errors import AmbiguousColumnError, ColumnMappingError, DataImportError, MissingColumnError, OpenECEError
from openece.io import ColumnRole, load_csv, resolve_columns
from openece.io.columns import BY_ELIMINATION, BY_NAME, EXPLICIT, TIME_NAMES

STEP_ROLES = (ColumnRole("time", "time", TIME_NAMES), ColumnRole("response", "response", by_elimination=True))


def resolve(path, explicit=None):
    return resolve_columns(load_csv(path), STEP_ROLES, explicit)


@pytest.mark.parametrize("header", ["time", "Time (s)", "timestamp", "t", "TIME [ms]", "elapsed_time", "time_s"])
def test_time_column_recognised_by_name(write_text, header):
    mapping = resolve(write_text("d.csv", f"{header},v\n0,1\n1,2\n2,3\n"))
    assert mapping["time"].column.name == header
    assert mapping["time"].method == BY_NAME
    assert mapping["response"].column.name == "v"
    assert mapping["response"].method == BY_ELIMINATION


def test_two_recognised_time_names_are_ambiguous(write_text):
    with pytest.raises(AmbiguousColumnError) as err:
        resolve(write_text("d.csv", "t,time,v\n0,0,1\n1,1,2\n"))
    assert err.value.role == "time"
    assert err.value.candidates == ("t", "time")
    assert "'t'" in str(err.value) and "'time'" in str(err.value)


def test_explicit_mapping_resolves_ambiguity_without_dropping_columns_silently(write_text):
    path = write_text("d.csv", "t,time,v\n0,0,1\n1,1,2\n")
    with pytest.raises(AmbiguousColumnError) as err:  # 't' and 'v' both remain as response candidates
        resolve(path, {"time": "time"})
    assert err.value.role == "response"
    mapping = resolve(path, {"time": "time", "response": "v"})
    assert mapping["time"].method == EXPLICIT and mapping["response"].method == EXPLICIT


def test_several_candidate_response_columns_are_ambiguous(write_text):
    with pytest.raises(AmbiguousColumnError) as err:
        resolve(write_text("d.csv", "time,command,speed\n0,0,0\n1,1,0.5\n"))
    assert err.value.candidates == ("command", "speed")
    assert err.value.hint == "response"


def test_missing_time_column_lists_what_is_available(write_text):
    with pytest.raises(MissingColumnError) as err:
        resolve(write_text("d.csv", "x,y\n0,1\n1,2\n"))
    message = str(err.value)
    assert "'x', 'y'" in message and "timestamp" in message
    assert err.value.role == "time"


def test_unknown_explicit_column_suggests_a_close_match(write_text):
    with pytest.raises(MissingColumnError, match="Did you mean 'speed'"):
        resolve(write_text("d.csv", "time,speed\n0,1\n1,2\n"), {"response": "sped"})


def test_explicit_lookup_ignores_case_and_units(write_text):
    mapping = resolve(write_text("d.csv", "Time (s),Speed (rpm),Current (A)\n0,1,2\n1,2,3\n"), {"response": "speed"})
    assert mapping["response"].column.name == "Speed (rpm)"


def test_explicit_non_numeric_column_is_rejected(write_text):
    with pytest.raises(DataImportError, match="not numeric"):
        resolve(write_text("d.csv", "time,label,v\n0,a,1\n1,b,2\n"), {"response": "label"})


def test_time_named_column_that_is_not_numeric_is_not_skipped_silently(write_text):
    with pytest.raises(ColumnMappingError, match="looks like the time column"):
        resolve(write_text("d.csv", "time,v\n0,1\nabc,2\n1,3\n"))


def test_one_column_cannot_fill_two_roles(write_text):
    with pytest.raises(ColumnMappingError, match="cannot be both"):
        resolve(write_text("d.csv", "time,v\n0,1\n1,2\n"), {"time": "v", "response": "v"})


def test_datetime_column_cannot_be_the_response(write_text):
    path = write_text("d.csv", "timestamp,v\n2026-01-01T00:00:00,1\n2026-01-01T00:00:01,2\n")
    with pytest.raises(ColumnMappingError, match="date-time"):
        resolve(path, {"time": "v", "response": "timestamp"})


def test_unknown_role_is_rejected(write_text):
    with pytest.raises(ColumnMappingError, match="unknown column role"):
        resolve(write_text("d.csv", "time,v\n0,1\n1,2\n"), {"voltage": "v"})


def test_mapping_errors_are_user_facing_not_key_errors(write_text):
    with pytest.raises(OpenECEError) as err:
        resolve(write_text("d.csv", "a,b,c\n1,2,3\n"))
    assert not isinstance(err.value, KeyError)
    assert isinstance(err.value, ValueError)


def test_single_letter_names_match_case_sensitively(write_text):
    assert resolve(write_text("a.csv", "t,v\n0,1\n1,2\n"))["time"].column.name == "t"
    with pytest.raises(MissingColumnError):  # 'T' is usually a temperature, never silently time
        resolve(write_text("b.csv", "T (degC),v\n20,1\n21,2\n"))


def test_an_unusable_optional_column_does_not_block(write_text):
    roles = (ColumnRole("x", "x", ("x",)), ColumnRole("phase", "phase", ("phase",), required=False))
    mapping = resolve_columns(load_csv(write_text("p.csv", "x,phase\n1,-3°\n2,-5°\n")), roles)
    assert "phase" not in mapping
    required = (ColumnRole("x", "x", ("x",)), ColumnRole("phase", "phase", ("phase",)))
    with pytest.raises(ColumnMappingError, match="cannot be used"):
        resolve_columns(load_csv(write_text("q.csv", "x,phase\n1,-3°\n2,-5°\n")), required)


def test_long_candidate_lists_are_shortened_and_explain_the_layout(write_text):
    row = ",".join(str(i) for i in range(50))
    with pytest.raises(AmbiguousColumnError) as err:
        resolve(write_text("row.csv", f"{row}\n"), {"time": "column_1"})
    assert "and 39 more" in str(err.value) and "transpose" in str(err.value)
