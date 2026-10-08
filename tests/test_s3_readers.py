import gzip
import io
from unittest.mock import MagicMock, patch

import pandas as pd
import polars as pl
import pytest

from ml_analytics.s3_connector import S3Connector


@pytest.fixture
def mock_s3():
    with patch("ml_analytics.s3_connector.boto3") as mock_boto3:
        mock_client = MagicMock()
        mock_boto3.Session.return_value.client.return_value = mock_client
        yield mock_client


def _rows(frame):
    if isinstance(frame, pl.DataFrame):
        return frame.to_dicts()
    return frame.to_dict(orient="records")


def _read(s3, method, payload, path, **kwargs):
    with patch.object(s3, "_download_s3_to_buffer", return_value=io.BytesIO(payload)) as download:
        result = getattr(s3, method)(path, **kwargs)
    return result, download


def test_get_path_strips_file_suffix_slash(mock_s3):
    s3 = S3Connector(bucket="test-bucket", s3_root="root")

    assert s3.get_path("events.json") == "s3://test-bucket/root/events.json"
    assert s3.get_path("events.json.gz") == "s3://test-bucket/root/events.json.gz"
    assert s3.get_path("events.csv") == "s3://test-bucket/root/events.csv"
    assert s3.get_path("events.csv.gz") == "s3://test-bucket/root/events.csv.gz"
    assert s3.get_path("events.parquet") == "s3://test-bucket/root/events.parquet"
    assert s3.get_path("events") == "s3://test-bucket/root/events/"


def test_read_json_array_returns_pandas(mock_s3):
    s3 = S3Connector(bucket="test-bucket", s3_root="projects")
    payload = b'[{"id": 1, "name": "a"}, {"id": 2, "name": "b"}]'

    result, download = _read(s3, "read_json", payload, "events.json")

    assert isinstance(result, pd.DataFrame)
    assert _rows(result) == [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}]
    download.assert_called_once_with("test-bucket", "projects/events.json")


def test_read_json_array_returns_polars(mock_s3):
    s3 = S3Connector(bucket="test-bucket")
    payload = b'[{"id": 1, "name": "a"}]'

    result, _download = _read(s3, "read_json", payload, "events.json", to_polars=True)

    assert isinstance(result, pl.DataFrame)
    assert _rows(result) == [{"id": 1, "name": "a"}]


def test_read_json_object_is_one_row(mock_s3):
    s3 = S3Connector(bucket="test-bucket")
    payload = b'{"id": 7, "name": "solo"}'

    result, _download = _read(s3, "read_json", payload, "one.json")

    assert _rows(result) == [{"id": 7, "name": "solo"}]


def test_read_json_lines_and_ndjson(mock_s3):
    s3 = S3Connector(bucket="test-bucket")
    payload = b'{"id": 1, "name": "a"}\n{"id": 2, "name": "b"}\n'

    jsonl, _download = _read(s3, "read_json", payload, "events.jsonl")
    ndjson, _download = _read(s3, "read_json", payload, "events.ndjson")

    assert _rows(jsonl) == [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}]
    assert _rows(ndjson) == _rows(jsonl)


def test_read_json_concatenated_documents(mock_s3):
    s3 = S3Connector(bucket="test-bucket")
    payload = b'{"id": 1}\n{"id": 2}'

    result, _download = _read(s3, "read_json", payload, "events.json")

    assert _rows(result) == [{"id": 1}, {"id": 2}]


def test_read_json_lines_false_rejects_extra_document(mock_s3):
    s3 = S3Connector(bucket="test-bucket")
    payload = b'{"id": 1}\n{"id": 2}'

    with pytest.raises(ValueError, match="Invalid JSON"):
        _read(s3, "read_json", payload, "events.json", lines=False)


def test_read_json_gzip(mock_s3):
    s3 = S3Connector(bucket="test-bucket")
    payload = gzip.compress(b'[{"id": 1, "name": "a"}]')

    result, download = _read(s3, "read_json", payload, "events.json.gz")

    assert _rows(result) == [{"id": 1, "name": "a"}]
    download.assert_called_once_with("test-bucket", "events.json.gz")


def test_read_json_directory_concatenates_and_skips_other_files(mock_s3):
    s3 = S3Connector(bucket="test-bucket")
    paginator = MagicMock()
    paginator.paginate.return_value = [
        {
            "Contents": [
                {"Key": "events/part-1.json"},
                {"Key": "events/notes.txt"},
                {"Key": "events/part-2.jsonl"},
            ]
        }
    ]
    mock_s3.get_paginator.return_value = paginator
    payloads = {
        "events/part-1.json": b'[{"id": 1, "name": "a"}, {"id": 2, "name": "b"}]',
        "events/part-2.jsonl": b'{"id": 3, "name": "c"}\n',
    }

    def download(_bucket, key):
        return io.BytesIO(payloads[key])

    with patch.object(s3, "_download_s3_to_buffer", side_effect=download) as mock_download:
        result = s3.read_json("events")

    assert _rows(result) == [
        {"id": 1, "name": "a"},
        {"id": 2, "name": "b"},
        {"id": 3, "name": "c"},
    ]
    assert [call.args[1] for call in mock_download.call_args_list] == [
        "events/part-1.json",
        "events/part-2.jsonl",
    ]
    paginator.paginate.assert_called_once_with(Bucket="test-bucket", Prefix="events/")


def test_read_json_directory_returns_polars(mock_s3):
    s3 = S3Connector(bucket="test-bucket")
    paginator = MagicMock()
    paginator.paginate.return_value = [
        {
            "Contents": [
                {"Key": "events/a.json"},
                {"Key": "events/b.json"},
            ]
        }
    ]
    mock_s3.get_paginator.return_value = paginator
    payloads = {
        "events/a.json": b'{"id": 1, "name": "a"}',
        "events/b.json": b'{"id": 2, "city": "x"}',
    }

    with patch.object(s3, "_download_s3_to_buffer", side_effect=lambda _bucket, key: io.BytesIO(payloads[key])):
        result = s3.read_json("events", to_polars=True)

    assert isinstance(result, pl.DataFrame)
    assert result["id"].to_list() == [1, 2]
    assert result["name"].to_list()[1] is None
    assert result["city"].to_list()[0] is None


def test_read_json_reports_bad_line(mock_s3):
    s3 = S3Connector(bucket="test-bucket")
    payload = b'{"id": 1}\n{bad}\n'

    with pytest.raises(ValueError, match="line 2"):
        _read(s3, "read_json", payload, "events.jsonl")


def test_read_json_missing_path_and_empty_prefix(mock_s3):
    s3 = S3Connector(bucket="test-bucket")

    with pytest.raises(ValueError, match="No file_path provided"):
        s3.read_json("")

    paginator = MagicMock()
    paginator.paginate.return_value = [{"Contents": [{"Key": "events/notes.txt"}]}]
    mock_s3.get_paginator.return_value = paginator
    with pytest.raises(ValueError, match="No JSON files found"):
        s3.read_json("events")


def test_read_csv_returns_pandas(mock_s3):
    s3 = S3Connector(bucket="test-bucket", s3_root="projects")
    payload = b'\xef\xbb\xbfid,name\n1,"a,b"\n2,c\n'

    result, download = _read(s3, "read_csv", payload, "events.csv")

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["id", "name"]
    assert _rows(result) == [{"id": 1, "name": "a,b"}, {"id": 2, "name": "c"}]
    download.assert_called_once_with("test-bucket", "projects/events.csv")


def test_read_csv_returns_polars(mock_s3):
    s3 = S3Connector(bucket="test-bucket")
    payload = b"id,name\n1,a\n2,b\n"

    result, _download = _read(s3, "read_csv", payload, "events.csv", to_polars=True)

    assert isinstance(result, pl.DataFrame)
    assert _rows(result) == [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}]


def test_read_csv_gzip_and_custom_separator(mock_s3):
    s3 = S3Connector(bucket="test-bucket")
    gzipped = gzip.compress(b"id,name\n1,a\n")
    piped = b"id|name\n1|a\n"

    gzip_result, _download = _read(s3, "read_csv", gzipped, "events.csv.gz")
    piped_result, _download = _read(s3, "read_csv", piped, "events.csv", sep="|")

    assert _rows(gzip_result) == [{"id": 1, "name": "a"}]
    assert _rows(piped_result) == [{"id": 1, "name": "a"}]


def test_read_csv_gzip_magic_without_gz_suffix(mock_s3):
    s3 = S3Connector(bucket="test-bucket")
    payload = gzip.compress(b"id,name\n4,d\n")

    result, _download = _read(s3, "read_csv", payload, "events.csv")

    assert _rows(result) == [{"id": 4, "name": "d"}]


def test_read_csv_directory_concatenates_and_aligns_columns(mock_s3):
    s3 = S3Connector(bucket="test-bucket")
    paginator = MagicMock()
    paginator.paginate.return_value = [
        {
            "Contents": [
                {"Key": "events/part-1.csv"},
                {"Key": "events/readme.txt"},
                {"Key": "events/part-2.csv"},
            ]
        }
    ]
    mock_s3.get_paginator.return_value = paginator
    payloads = {
        "events/part-1.csv": b"id,name\n1,a\n",
        "events/part-2.csv": b"id,city\n2,x\n",
    }

    with patch.object(s3, "_download_s3_to_buffer", side_effect=lambda _bucket, key: io.BytesIO(payloads[key])):
        pandas_result = s3.read_csv("events")
        polars_result = s3.read_csv("events", to_polars=True)

    assert list(pandas_result["id"]) == [1, 2]
    assert pandas_result.loc[0, "name"] == "a"
    assert pd.isna(pandas_result.loc[1, "name"])
    assert pandas_result.loc[1, "city"] == "x"
    assert isinstance(polars_result, pl.DataFrame)
    assert polars_result["id"].to_list() == [1, 2]
    assert polars_result["city"].to_list()[0] is None
    paginator.paginate.assert_called_with(Bucket="test-bucket", Prefix="events/")


def test_read_csv_rejects_bad_input(mock_s3):
    s3 = S3Connector(bucket="test-bucket")

    with pytest.raises(ValueError, match="No file_path provided"):
        s3.read_csv("")
    with pytest.raises(ValueError, match="sep must be a single character"):
        s3.read_csv("events.csv", sep="||")
    with pytest.raises(ValueError, match="empty"):
        _read(s3, "read_csv", b"   \n", "events.csv")

    paginator = MagicMock()
    paginator.paginate.return_value = [{"Contents": [{"Key": "events/notes.txt"}]}]
    mock_s3.get_paginator.return_value = paginator
    with pytest.raises(ValueError, match="No CSV files found"):
        s3.read_csv("events")
