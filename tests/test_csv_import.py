import hashlib

import numpy as np
import pytest

from openece.errors import DataImportError, OpenECEError
from openece.io import load_csv


def test_header_units_comments_and_provenance(write_text):
    path = write_text("scope.csv", "# SYNTHETIC test data\n# second comment\nTime (ms),V(out),speed_rpm\n0,1.5,10\n1,2.5,20\n2,3.5,30\n")
    ds = load_csv(path)
    assert ds.column_names == ["Time (ms)", "V(out)", "speed_rpm"]
    assert [c.unit for c in ds.columns] == ["ms", None, "rpm"]
    np.testing.assert_array_equal(ds.find_column("V(out)").values, [1.5, 2.5, 3.5])
    assert ds.metadata["comments"] == ["SYNTHETIC test data", "second comment"]
    assert ds.metadata["header_line"] == 3
    assert ds.metadata["first_data_line"] == 4
    assert ds.metadata["delimiter"] == "comma"
    assert ds.locate(0) == "line 4"
    assert ds.sample_rate_hz is None
    assert ds.source.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert ds.source.size_bytes == path.stat().st_size


@pytest.mark.parametrize("text, delimiter", [
    ("a;b\n1;2\n3;4\n", "semicolon"),
    ("a\tb\n1\t2\n3\t4\n", "tab"),
    ("a b\n1 2\n3 4\n", "whitespace"),
    ("a,b\r\n1,2\r\n3,4\r\n", "comma"),
    ("a,b\r1,2\r3,4\r", "comma"),
])
def test_delimiter_and_line_ending_detection(write_text, text, delimiter):
    ds = load_csv(write_text("data.csv", text))
    assert ds.metadata["delimiter"] == delimiter
    np.testing.assert_array_equal(ds.find_column("b").values, [2.0, 4.0])


def test_explicit_delimiter(write_text):
    ds = load_csv(write_text("data.txt", "a|b\n1|2\n"), delimiter="|")
    assert ds.column_names == ["a", "b"]


def test_decimal_comma_needs_an_explicit_option(write_text):
    path = write_text("euro.csv", "zeit;wert\n0,1;1,5\n0,2;2,5\n")
    with pytest.raises(DataImportError, match="decimal comma") as err:
        load_csv(path)
    assert err.value.hint == "decimal"
    ds = load_csv(path, decimal=",")
    np.testing.assert_allclose(ds.find_column("wert").values, [1.5, 2.5])
    assert ds.metadata["decimal"] == ","


def test_headerless_file_gets_numbered_columns(write_text):
    ds = load_csv(write_text("raw.txt", "0 1.0\n1 2.0\n2 4.0\n"))
    assert ds.column_names == ["column_1", "column_2"]
    assert ds.metadata["header_line"] is None


def test_picoscope_style_units_row(write_text):
    ds = load_csv(write_text("pico.csv", "Time,Channel A\r\n(us),(V)\r\n\r\n-1.0,0.5\r\n0.0,0.6\r\n1.0,0.7\r\n"))
    assert [c.unit for c in ds.columns] == ["us", "V"]
    assert ds.metadata["units_line"] == 2
    assert ds.metadata["first_data_line"] == 4
    assert any("units row" in w for w in ds.warnings)


def test_utf8_bom_is_removed(write_text):
    ds = load_csv(write_text("bom.csv", "﻿time,value\n0,1\n1,2\n"))
    assert ds.column_names == ["time", "value"]


def test_latin1_fallback_is_reported(write_text):
    ds = load_csv(write_text("latin.csv", "phase (°),v\n1,2\n", encoding="latin-1"))
    assert ds.columns[0].unit == "°"
    assert ds.metadata["encoding"] == "latin-1"
    assert any("Latin-1" in w for w in ds.warnings)


def test_trailing_delimiters_and_empty_rows_are_ignored(write_text):
    ds = load_csv(write_text("trail.csv", "t,v,\n0,1,\n\n1,2,\n,,\n"))
    assert ds.column_names == ["t", "v"]
    assert ds.n_rows == 2


def test_text_and_mixed_columns_are_reported_not_coerced(write_text):
    ds = load_csv(write_text("mixed.csv", "time,v,label\n0,1,a\n1,x,b\n2,3,c\n"))
    assert ds.column_names == ["time"]
    reasons = {s.name: s.reason for s in ds.skipped}
    assert "line 3" in reasons["v"] and "'x'" in reasons["v"]
    assert "not numeric" in reasons["label"]
    with pytest.raises(DataImportError, match="line 3"):
        ds.find_column("v")


def test_blank_cells_are_tracked(write_text):
    ds = load_csv(write_text("gaps.csv", "t,v\n0,1\n1,\n2,3\n"))
    v = ds.find_column("v")
    assert v.blank_rows == (1,)
    assert np.isnan(v.values[1])


def test_iso_timestamps_become_seconds(write_text):
    ds = load_csv(write_text("log.csv", "timestamp,temp (degC)\n2026-09-26T10:00:00Z,20.1\n2026-09-26T10:00:01.5Z,20.2\n"))
    stamp = ds.find_column("timestamp")
    assert stamp.kind == "datetime" and stamp.unit == "s"
    np.testing.assert_allclose(stamp.values, [0.0, 1.5])
    assert "2026-09-26T10:00:00+00:00" in stamp.note


def test_duplicate_headers_are_renamed_with_a_warning(write_text):
    ds = load_csv(write_text("dup.csv", "v,v\n1,2\n3,4\n"))
    assert ds.column_names == ["v", "v_2"]
    assert any("duplicate" in w for w in ds.warnings)


def test_extra_fields_are_an_error_with_line_number(write_text):
    with pytest.raises(DataImportError, match="line 3 has 3 fields"):
        load_csv(write_text("ragged.csv", "a,b\n1,2\n3,4,5\n"))


def test_preamble_needs_skip_rows(write_text):
    path = write_text("preamble.csv", "Model,XYZ-100\nSerial,12345\ntime,v\n0,1\n1,2\n")
    with pytest.raises(DataImportError, match="preamble") as err:
        load_csv(path)
    assert err.value.hint == "skip_rows"
    ds = load_csv(path, skip_rows=2)
    assert ds.column_names == ["time", "v"]
    assert ds.metadata["header_line"] == 3


@pytest.mark.parametrize("text, message", [
    ("", "no data"),
    ("# only a comment\n\n", "no data"),
    ("a,b\n", "no data rows"),
    ('a,b\n"1,2\n3,4\n', "unbalanced quote"),
])
def test_empty_or_malformed_files(write_text, text, message):
    with pytest.raises(DataImportError, match=message):
        load_csv(write_text("bad.csv", text))


def test_binary_file_is_rejected(tmp_path):
    path = tmp_path / "binary.csv"
    path.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt ")
    with pytest.raises(DataImportError, match="text file"):
        load_csv(path)


def test_missing_file(tmp_path):
    with pytest.raises(DataImportError, match="not found"):
        load_csv(tmp_path / "nope.csv")


@pytest.mark.parametrize("options", [
    {"delimiter": "::"}, {"decimal": ";"}, {"skip_rows": -1}, {"delimiter": ",", "decimal": ","}, {"encoding": "no-such-codec"},
])
def test_invalid_options_are_user_errors(write_text, options):
    with pytest.raises(OpenECEError):
        load_csv(write_text("ok.csv", "a,b\n1,2\n"), **options)


def test_row_vector_file_does_not_crash_the_delimiter_detection(write_text):
    ds = load_csv(write_text("row.csv", ",".join(f"{i * 1e-3:.6f}" for i in range(20_000)) + "\n"))
    assert len(ds.columns) == 20_000 and ds.n_rows == 1


def test_oversized_field_error_names_its_line(write_text):
    text = "time,voltage\n0,1\n1,2\n2,3\n3," + "9" * 200_000 + "\n4,5\n"
    with pytest.raises(DataImportError, match=r"line 5: .*field limit") as err:
        load_csv(write_text("huge.csv", text))
    assert err.value.hint == "delimiter"


def test_utf32_and_truncated_utf16(tmp_path):
    utf32 = tmp_path / "u32.csv"
    utf32.write_bytes("time,v\n0,1\n1,2\n".encode("utf-32"))
    ds = load_csv(utf32)
    assert ds.column_names == ["time", "v"] and ds.metadata["encoding"] == "utf-32"
    broken = tmp_path / "u16.csv"
    broken.write_bytes("time,v\n0,1\n".encode("utf-16") + b"\x31")  # odd number of bytes
    with pytest.raises(DataImportError, match="UTF-16") as err:
        load_csv(broken)
    assert err.value.hint == "encoding"


def test_semicolon_split_wins_over_decimal_commas(write_text):
    path = write_text("eu.csv", "0,000;2,000\n0,001;2,309\n0,002;2,500\n")
    with pytest.raises(DataImportError, match="decimal comma"):
        load_csv(path)  # never mis-split on the decimal commas
    ds = load_csv(path, decimal=",")
    assert ds.metadata["delimiter"] == "semicolon" and ds.metadata["header_line"] is None
    np.testing.assert_allclose(ds.columns[1].values, [2.0, 2.309, 2.5])


def test_headerless_integer_pairs_warn_about_a_possible_decimal_comma(write_text):
    ds = load_csv(write_text("ints.csv", "2,0000\n2,3090\n2,5100\n"))
    assert any("decimal comma" in w for w in ds.warnings)
    assert not load_csv(write_text("floats.csv", "0.5,1.5\n1.5,2.5\n")).warnings


def test_bom_is_removed_with_an_explicit_encoding(write_text):
    ds = load_csv(write_text("bom.csv", "﻿0.5,1.5\n1.5,2.5\n"), encoding="utf-8")
    assert ds.metadata["header_line"] is None and ds.n_rows == 2


def test_quote_character_is_not_a_valid_delimiter(write_text):
    with pytest.raises(DataImportError, match="invalid delimiter"):
        load_csv(write_text("q.csv", "a,b\n1,2\n"), delimiter='"')
