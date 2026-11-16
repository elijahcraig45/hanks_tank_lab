import importlib.util
import subprocess
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "publish_public", Path(__file__).resolve().parents[1] / "scripts" / "publish_public.py")
pp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pp)


def _rules(tmp_path, text):
    path = tmp_path / "public.rules"
    path.write_text(text)
    return path


def test_parse_rules_sorts_replacements_longest_first(tmp_path):
    replace, deny, exclude = pp.parse_rules(_rules(tmp_path, (
        "# comment\n"
        "replace host => <host>\n"
        "replace host.example.net => <host>.example.net\n"
        "deny secret\n"
        "exclude private/\n")))
    assert replace[0][0] == "host.example.net"
    assert deny == ["secret"] and exclude == ["private/"]


def test_parse_rules_rejects_unknown_lines(tmp_path):
    with pytest.raises(SystemExit):
        pp.parse_rules(_rules(tmp_path, "rename a => b\n"))


def test_clean_replaces_text_and_drops_excluded(tmp_path):
    tree = tmp_path / "tree"
    (tree / "private").mkdir(parents=True)
    (tree / "a.md").write_text("ssh host.example.net as me@personal.example\n")
    (tree / "private" / "notes.md").write_text("x")
    changed = pp.clean(tree, [("host.example.net", "<host>"), ("me@personal.example", "you@example.com")], ["private/"])
    assert changed == 1
    assert (tree / "a.md").read_text() == "ssh <host> as you@example.com\n"
    assert not (tree / "private" / "notes.md").exists()


# Built from pieces so this file itself passes the leak check it tests.
@pytest.mark.parametrize("line", [
    "url = http://localhost:3001/api/push/" + "AbC123xyZ9?status=up",
    "host = homelab.tail" + "1a2b3c.ts.net",
    "lan = 192.168." + "1.20",
    "tailnet = 100.101." + "102.103",
    "key = AIza" + "S" * 35,
])
def test_builtin_deny_catches_private_values(tmp_path, line):
    (tmp_path / "f.txt").write_text(line + "\n")
    assert pp.leaks(tmp_path, [])


@pytest.mark.parametrize("line", [
    "url = http://localhost:3001/api/push/<push-token>?status=up",
    "host = homelab.<tailnet>.ts.net",
    "key = AIza" + "FakeKeyForTheRedactionTest" + "0" * 9,
])
def test_placeholders_pass(tmp_path, line):
    (tmp_path / "f.txt").write_text(line + "\n")
    assert pp.leaks(tmp_path, []) == []


def test_local_deny_rules_apply_and_report_no_matched_text(tmp_path):
    (tmp_path / "f.txt").write_text("works at Example Corp\n")
    found = pp.leaks(tmp_path, ["example corp"])
    assert found and "Example Corp" not in found[0]


def test_refuses_rules_inside_the_repo_and_a_dest_without_git(tmp_path):
    repo = Path(pp.__file__).resolve().parents[1]
    assert pp.main(["--rules", str(repo / "README.md"), str(tmp_path)]) == 2
    rules = _rules(tmp_path, "deny nothing-matches-this\n")
    assert pp.main(["--rules", str(rules), str(tmp_path / "no-git")]) == 2


def test_sync_keeps_dest_git_and_replaces_the_rest(tmp_path):
    tree, dest = tmp_path / "tree", tmp_path / "dest"
    (tree / "src").mkdir(parents=True)
    (tree / "src" / "a.py").write_text("new")
    (dest / ".git").mkdir(parents=True)
    (dest / ".git" / "HEAD").write_text("ref")
    (dest / "stale.txt").write_text("old")
    pp.sync(tree, dest)
    assert (dest / ".git" / "HEAD").read_text() == "ref"
    assert (dest / "src" / "a.py").read_text() == "new" and not (dest / "stale.txt").exists()
