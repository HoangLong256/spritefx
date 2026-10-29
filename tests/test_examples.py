"""Every shipped example builds with zero faults: the examples are the quality bar."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fx.__main__ import main

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
SPECS = sorted(EXAMPLES.rglob("*.json"))  # the demo scene's effects live in a subfolder


@pytest.mark.parametrize("spec", SPECS, ids=lambda p: p.stem)
def test_example_builds_with_zero_faults(spec: Path, tmp_path: Path) -> None:
    name = spec.stem
    assert main(["build", str(spec), "--out", str(tmp_path)]) == 0
    sheet = json.loads((tmp_path / name / "sheets" / f"{name}.json").read_text())
    assert sheet["frames"] and sheet["anchor"]


def test_examples_render_byte_identical(tmp_path: Path) -> None:
    for run in ("a", "b"):
        assert main(["build", str(EXAMPLES / "hit_spark.json"), "--out", str(tmp_path / run)]) == 0
    a = (tmp_path / "a" / "hit_spark" / "sheets" / "hit_spark.png").read_bytes()
    b = (tmp_path / "b" / "hit_spark" / "sheets" / "hit_spark.png").read_bytes()
    assert a == b
