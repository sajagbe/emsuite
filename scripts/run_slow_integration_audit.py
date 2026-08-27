#!/usr/bin/env python3
"""Run all slow integration tests with full artifact traceability."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = REPO_ROOT / "tests" / "integration_runs"


def _git_info() -> dict[str, str | None]:
    def run(cmd: list[str]) -> str | None:
        try:
            return subprocess.check_output(
                cmd, cwd=REPO_ROOT, stderr=subprocess.DEVNULL, text=True
            ).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    return {
        "commit": run(["git", "rev-parse", "HEAD"]),
        "branch": run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "describe": run(["git", "describe", "--tags", "--always", "--dirty"]),
    }


def _emsuite_version() -> str:
    try:
        import emsuite

        return emsuite.__version__
    except Exception:
        return "unknown"


def _write_run_manifest(run_dir: Path) -> None:
    lines = [
        "# EMSuite slow integration audit",
        "",
        f"- **Run id:** `{run_dir.name}`",
        f"- **Completed (UTC):** {datetime.now(UTC).isoformat()}",
        "",
        "## Test traceability",
        "",
        "| Status | Feature | Version | Channel | Test | Duration (s) |",
        "|--------|---------|---------|---------|------|--------------|",
    ]

    passed = failed = skipped = 0
    for test_dir in sorted(run_dir.iterdir()):
        if not test_dir.is_dir():
            continue
        manifest_path = test_dir / "manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text())
        outcome = manifest.get("outcome", "?")
        if outcome == "passed":
            passed += 1
            icon = "PASS"
        elif outcome == "skipped":
            skipped += 1
            icon = "SKIP"
        else:
            failed += 1
            icon = "FAIL"

        feature = manifest.get("feature_id", test_dir.name)
        version = manifest.get("version", "—")
        channel = manifest.get("channel", "—")
        nodeid = manifest.get("nodeid", test_dir.name)
        duration = manifest.get("duration_s", "—")
        summary = manifest.get("summary", "")
        lines.append(f"| {icon} | `{feature}` | {version} | {channel} | `{nodeid}` | {duration} |")
        if summary:
            lines.append(f"| | | | | _{summary}_ | |")

    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- Passed: **{passed}**",
            f"- Failed: **{failed}**",
            f"- Skipped: **{skipped}**",
            "",
            "## Per-test artifacts",
            "",
            "Each test subdirectory contains:",
            "",
            "- `manifest.json` — feature mapping and outcome",
            "- `workspace/` — full test working directory (inputs, outputs, logs)",
            "- `inputs/` — copies of `*.in` and `*.xyz`",
            "- `assertions.json` — checks recorded by the test",
            "- `failure.txt` — present when a test fails",
            "",
        ]
    )

    (run_dir / "RUN_MANIFEST.md").write_text("\n".join(lines))


def main() -> int:
    run_id = datetime.now(UTC).strftime("%Y-%m-%dT%H%M%SZ")
    run_dir = RUNS_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    run_dir = run_dir.resolve()

    meta = {
        "run_id": run_id,
        "started_utc": datetime.now(UTC).isoformat(),
        "emsuite_version": _emsuite_version(),
        "git": _git_info(),
        "pytest_command": [
            sys.executable,
            "-m",
            "pytest",
            "tests/integration",
            "-v",
            "-m",
            "slow",
        ],
    }
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))

    env = os.environ.copy()
    env["EMSUITE_INTEGRATION_RUN_DIR"] = str(run_dir)
    env["PYTHONPATH"] = str(REPO_ROOT / "src") + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )

    cmd = [
        *meta["pytest_command"],
        f"--junitxml={run_dir / 'junit.xml'}",
        f"--log-file={run_dir / 'pytest.log'}",
        "--log-file-level=INFO",
    ]

    print(f"Integration audit run: {run_dir}")
    print("Running:", " ".join(cmd))

    result = subprocess.run(cmd, cwd=REPO_ROOT, env=env)
    meta["finished_utc"] = datetime.now(UTC).isoformat()
    meta["exit_code"] = result.returncode
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))

    _write_run_manifest(run_dir)
    latest_link = RUNS_ROOT / "LATEST"
    if latest_link.is_symlink() or latest_link.exists():
        latest_link.unlink()
    latest_link.symlink_to(run_dir.name)
    print(f"\nManifest written: {run_dir / 'RUN_MANIFEST.md'}")
    print(f"Latest symlink: {latest_link} -> {run_dir.name}")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
