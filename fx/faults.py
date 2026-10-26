"""Fault records and the fixed-column report format shared by every audit."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Fault:
    check: str
    subject: str
    where: str = ""
    detail: str = ""

    def line(self) -> str:
        return f"FAIL {self.check:<13} {self.subject:<21} {self.where:<9} {self.detail}".rstrip()


def format_report(audit: str, target: str, faults: list[Fault], as_json: bool = False) -> str:
    if as_json:
        return json.dumps(
            {"audit": audit, "target": target, "faults": [asdict(f) for f in faults], "count": len(faults)},
            indent=2,
        )
    lines = [f"{audit}: {target}"]
    lines += [f.line() for f in faults]
    lines.append(f"{audit}: {len(faults)} faults")
    return "\n".join(lines)
