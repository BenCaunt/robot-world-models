from __future__ import annotations

import json
import subprocess

from robot_world_models.warmhub import WarmHubCLI, normalize_wref


def test_normalize_wref_strips_only_numeric_version_pin() -> None:
    assert normalize_wref("Robot/so-arm101@v3") == "Robot/so-arm101"
    assert normalize_wref("Robot/so-arm101@version") == "Robot/so-arm101@version"
    assert normalize_wref("Robot/so-arm101") == "Robot/so-arm101"


def test_discover_queries_both_first_class_repositories(monkeypatch) -> None:
    commands: list[list[str]] = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        repo = command[command.index("--repo") + 1]
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps({"items": [{"repo": repo}]}),
            stderr="",
        )

    monkeypatch.setattr("shutil.which", lambda _name: "/usr/local/bin/wh")
    monkeypatch.setattr("subprocess.run", fake_run)

    result = WarmHubCLI().discover("SO-101")

    assert result["models"] == [{"repo": "bencaunt/robot-models"}]
    assert result["datasets"] == [{"repo": "bencaunt/robot-datasets"}]
    assert [command[command.index("--repo") + 1] for command in commands] == [
        "bencaunt/robot-models",
        "bencaunt/robot-datasets",
    ]
    assert all("--mode" in command and "hybrid" in command for command in commands)

