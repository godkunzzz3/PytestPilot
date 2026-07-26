from pathlib import Path

from firstcoder.tools.builtin import create_builtin_registry
from firstcoder.tools.fresh_source import FreshSourceGuard


def _guarded_tools(root: Path, editable_paths: list[str]):
    guard = FreshSourceGuard(root, editable_paths=editable_paths)
    registry = create_builtin_registry(
        root,
        include_mutation_tools=True,
        fresh_source_guard=guard,
    )
    return {tool.name: tool for tool in registry.tools()}


def test_edit_requires_token_for_the_same_path(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("A = 1\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("B = 1\n", encoding="utf-8")
    tools = _guarded_tools(tmp_path, ["a.py", "b.py"])

    revision = tools["view"].executor(path="a.py").data["source_revision"]
    missing = tools["edit"].executor(path="b.py", old="B = 1", new="B = 2")
    wrong = tools["edit"].executor(
        path="b.py",
        old="B = 1",
        new="B = 2",
        read_token=revision["token"],
    )

    assert missing.ok is False
    assert "read_token" in missing.error
    assert wrong.ok is False
    assert "belongs to a.py" in wrong.error
    assert (tmp_path / "b.py").read_text(encoding="utf-8") == "B = 1\n"


def test_edit_rejects_stale_hash_and_does_not_mutate(tmp_path: Path) -> None:
    target = tmp_path / "a.py"
    target.write_text("A = 1\n", encoding="utf-8")
    tools = _guarded_tools(tmp_path, ["a.py"])
    revision = tools["view"].executor(path="a.py").data["source_revision"]
    target.write_text("A = 3\n", encoding="utf-8")

    result = tools["edit"].executor(
        path="a.py",
        old="A = 3",
        new="A = 2",
        read_token=revision["token"],
    )

    assert result.ok is False
    assert "source changed after read" in result.error
    assert target.read_text(encoding="utf-8") == "A = 3\n"


def test_successful_edit_consumes_token(tmp_path: Path) -> None:
    target = tmp_path / "a.py"
    target.write_text("A = 1\n", encoding="utf-8")
    tools = _guarded_tools(tmp_path, ["a.py"])
    revision = tools["view"].executor(path="a.py").data["source_revision"]

    first = tools["edit"].executor(
        path="a.py",
        old="A = 1",
        new="A = 2",
        read_token=revision["token"],
    )
    replay = tools["edit"].executor(
        path="a.py",
        old="A = 2",
        new="A = 3",
        read_token=revision["token"],
    )

    assert first.ok is True
    assert replay.ok is False
    assert "unknown read_token" in replay.error


def test_new_file_must_be_inside_editable_paths(tmp_path: Path) -> None:
    tools = _guarded_tools(tmp_path, ["src/new.py"])

    denied = tools["write"].executor(path="other.py", content="x = 1\n")
    allowed = tools["write"].executor(path="src/new.py", content="x = 1\n")

    assert denied.ok is False
    assert "outside editable_paths" in denied.error
    assert allowed.ok is True


def test_read_multi_returns_per_file_revision_tokens(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("A = 1\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("B = 1\n", encoding="utf-8")
    tools = _guarded_tools(tmp_path, ["a.py", "b.py"])

    result = tools["read_multi"].executor(paths=["a.py", "b.py"])

    revisions = [item["source_revision"] for item in result.data["files"]]
    assert [revision["path"] for revision in revisions] == ["a.py", "b.py"]
    assert all(revision["token"] in result.content for revision in revisions)


def test_apply_patch_requires_token_for_each_existing_path(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("A = 1\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("B = 1\n", encoding="utf-8")
    tools = _guarded_tools(tmp_path, ["a.py", "b.py"])
    revision = tools["view"].executor(path="a.py").data["source_revision"]
    patch = (
        "*** Begin Patch\n"
        "*** Update File: a.py\n"
        "@@\n"
        "-A = 1\n"
        "+A = 2\n"
        "*** Update File: b.py\n"
        "@@\n"
        "-B = 1\n"
        "+B = 2\n"
        "*** End Patch"
    )

    denied = tools["apply_patch"].executor(
        patch=patch,
        read_tokens={"a.py": revision["token"]},
    )

    assert denied.ok is False
    assert "read_token" in denied.error
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "A = 1\n"
    assert (tmp_path / "b.py").read_text(encoding="utf-8") == "B = 1\n"
