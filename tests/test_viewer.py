import json
import threading
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import urlopen

import pytest

from rlllmf.viewer import export_viewer, make_viewer_server


def record(answer="first"):
    return (
        json.dumps(
            {
                "instruction": "task",
                "responses": [answer],
                "judgments": [{"id": "c0", "optimism": 0.5}],
            }
        )
        + "\n"
    )


@pytest.fixture
def live_server(tmp_path):
    root = tmp_path / "runs"
    root.mkdir()
    server = make_viewer_server(root, port=0)
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01))
    thread.start()
    try:
        yield root, f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def get(url):
    with urlopen(url, timeout=5) as response:
        assert response.headers["Cache-Control"] == "no-store"
        return json.load(response)


def test_live_server_discovers_new_runs_and_reads_appended_judgments(live_server):
    root, url = live_server
    assert get(url + "/api/logs")["logs"] == []
    first = root / "one" / "judge.jsonl"
    first.parent.mkdir()
    first.write_text(record())
    logs = get(url + "/api/logs")["logs"]
    assert [row["path"] for row in logs] == ["one/judge.jsonl"]
    assert get(url + "/api/log?path=one/judge.jsonl")["text"] == record()
    with first.open("a") as stream:
        stream.write(record("second"))
    assert get(url + "/api/logs")["logs"][0]["version"] != logs[0]["version"]
    assert len(get(url + "/api/log?path=one/judge.jsonl")["text"].splitlines()) == 2
    second = root / "two" / "judge.jsonl"
    second.parent.mkdir()
    second.write_text(record("another run"))
    assert len(get(url + "/api/logs")["logs"]) == 2
    assert "another run" not in get(url + "/api/log?path=one/judge.jsonl")["text"]
    report = root / "two" / "comparison.jsonl"
    report.write_text(record("comparison"))
    assert "two/comparison.jsonl" in [row["path"] for row in get(url + "/api/logs")["logs"]]


def test_partial_last_record_is_retried_without_losing_complete_records(live_server):
    root, url = live_server
    log = root / "judge.jsonl"
    following = record("second").rstrip()
    split = len(following) // 2
    log.write_text(record() + following[:split])
    result = get(url + "/api/log?path=judge.jsonl")
    assert result["pending"] is True
    assert result["text"] == record()
    with log.open("a") as stream:
        stream.write(following[split:])
    result = get(url + "/api/log?path=judge.jsonl")
    assert result["pending"] is False
    assert len(result["text"].splitlines()) == 2


def test_only_judge_logs_are_exposed_and_metadata_contains_no_secrets(live_server):
    root, url = live_server
    (root / "judge.jsonl").write_text(record())
    (root / "run.json").write_text(
        json.dumps(
            {
                "settings": {
                    "api": {"anthropic_api_key": "must-never-be-exposed"},
                    "training": {"model": "local-policy"},
                    "judge": {"model": "future-judge"},
                }
            }
        )
    )
    secret = root.parent / ".config"
    secret.write_text("must-never-be-exposed")
    (root / "memory.jsonl").write_text("{}\n")
    (root / "outside.judge.jsonl").symlink_to(secret)
    logs = get(url + "/api/logs")["logs"]
    assert [row["path"] for row in logs] == ["judge.jsonl"]
    payload = get(url + "/api/log?path=judge.jsonl")
    assert payload["metadata"] == {"policy_model": "local-policy", "judge_model": "future-judge"}
    assert "must-never-be-exposed" not in json.dumps(payload)
    for path in [
        "/.config",
        "/run.json",
        "/api/log?path=../.config",
        "/api/log?path=" + quote(str(secret)),
        "/api/log?path=outside.judge.jsonl",
    ]:
        with pytest.raises(HTTPError) as error:
            get(url + path)
        assert error.value.code == 404


def test_export_preserves_log_text_without_executable_html(tmp_path):
    attack = "</script><script>alert('not executable')</script>"
    log = tmp_path / "judge.jsonl"
    log.write_text(record(attack))
    page = export_viewer(log)
    html = page.read_text()
    assert attack not in html
    marker = '<script id="embedded-log" type="application/json">'
    payload = json.loads(html.split(marker)[1].split("</script>")[0])
    assert payload["text"] == log.read_text()
    with pytest.raises(ValueError):
        export_viewer(log, log)
