"""CLI: python -m fx <command> ..."""

from __future__ import annotations

import argparse
import json
import os
import sys
from importlib import metadata
from pathlib import Path

from .faults import Fault, format_report
from .palette import check_palette, expand_ramps, find_palette
from .spec import load_effect

PINNED = {"numpy": "2.4.6", "Pillow": "12.3.0", "jsonschema": "4.26.0"}


def _emit(text: str) -> None:
    print(text, flush=True)


def _counted(faults: list[Fault]) -> list[Fault]:
    """Faults that fail a command; warnings are printed but never fail it (`fx.audit.faults_only`)."""
    from .audit import faults_only

    return faults_only(faults)


def _load(args: argparse.Namespace, audit: str):
    effect, faults = load_effect(args.spec)
    if faults:
        _emit(format_report(audit, args.spec, faults, args.json))
    return effect


def _render(effect, args: argparse.Namespace) -> dict:
    from .pipeline import Settings, render_effect
    from .sheets import write_review

    summary = render_effect(effect, Path(args.out), Settings(supersample=args.supersample))
    write_review(effect, Path(args.out))
    warns = [line for line in summary["cleanup_log"] if "WARN" in line]
    if warns and not args.json:
        _emit(f"cleanup: {len(warns)} warnings (a layer lost over a quarter of its pixels); see render.json")
    return summary


def _audit(effect, args: argparse.Namespace) -> list[Fault]:
    from .audit import audit_fx, curve_sheet

    root = Path(args.out) / effect.name
    if not (root / "render.json").exists():
        return [Fault("render", effect.name, "", f"no render at {root}; run render or build first")]
    faults = audit_fx(effect, Path(args.out))
    curve_sheet(effect, Path(args.out))
    return faults


def cmd_validate(args: argparse.Namespace) -> int:
    effect = _load(args, "validate")
    if effect is None:
        return 1
    _emit(format_report("validate", args.spec, [], args.json))
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    effect = _load(args, "render")
    if effect is None:
        return 1
    summary = _render(effect, args)
    root = Path(args.out) / effect.name
    if args.json:
        _emit(json.dumps({"render": str(root), "frames": len(summary["frames"]), "canvas": summary["canvas"],
                          "anchor": summary["anchor"]}, indent=2))
    else:
        _emit(f"render: {effect.name} {len(summary['frames'])} frames {summary['canvas'][0]}x{summary['canvas'][1]} "
              f"-> {root}")
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    effect = _load(args, "audit")
    if effect is None:
        return 1
    faults = _audit(effect, args)
    _emit(format_report("audit", args.spec, faults, args.json))
    return 1 if _counted(faults) else 0


def cmd_build(args: argparse.Namespace) -> int:
    """validate -> render (with review sheets) -> audit -> game sheets; stops on faults."""
    from .sheets import write_sheets

    effect = _load(args, "build")
    if effect is None:
        return 1
    _render(effect, args)
    faults = _audit(effect, args)
    if _counted(faults):
        _emit(format_report("build", args.spec, faults, args.json))
        return 1
    written = write_sheets(effect, Path(args.out))
    _emit(format_report("build", args.spec, faults, args.json))
    if not args.json:
        _emit("sheets: " + ", ".join(str(p) for p in written))
    return 0


def cmd_gif(args: argparse.Namespace) -> int:
    from .sheets import write_gif

    effect = _load(args, "gif")
    if effect is None:
        return 1
    try:
        path = write_gif(effect, Path(args.out), zoom=args.zoom)
    except FileNotFoundError as exc:
        _emit(format_report("gif", args.spec, [Fault("render", effect.name, "", str(exc)[:160])], args.json))
        return 1
    _emit(f"gif: {path}")
    return 0


def cmd_palette_check(args: argparse.Namespace) -> int:
    try:
        path = find_palette(args.palette, Path.cwd())
        data = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        _emit(format_report("palette-check", args.palette, [Fault("palette", args.palette, "", str(exc)[:160])],
                            args.json))
        return 1
    faults = check_palette(data)
    _emit(format_report("palette-check", str(path), faults, args.json))
    if not faults and not args.json:
        for name, colors in expand_ramps(data)["ramps"].items():
            _emit(f"  {name:<12} {' '.join(colors)}")
    return 1 if faults else 0


def cmd_doctor(args: argparse.Namespace) -> int:
    faults = []
    if sys.version_info < (3, 11):
        faults.append(Fault("doctor", "python", "", f"need 3.11+, found {sys.version.split()[0]}"))
    for pkg, want in PINNED.items():
        try:
            have = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            have = None
        if have != want:
            faults.append(Fault("doctor", pkg, "", f"need {want}, found {have}"))
    out = Path(args.out).resolve()
    nearest = next(d for d in (out, *out.parents) if d.exists())  # the folder the output would be made in
    if not (nearest.is_dir() and os.access(nearest, os.W_OK | os.X_OK)):
        faults.append(Fault("doctor", str(out), "", f"cannot write: {nearest} is not a writable folder"))
    _emit(format_report("doctor", sys.executable, faults, args.json))
    if any(f.subject in PINNED for f in faults) and not args.json:
        _emit("install: python -m pip install " + " ".join(f"{k}=={v}" for k, v in PINNED.items()))
    return 1 if faults else 0


def _positive(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be 1 or more")
    return value


def main(argv: list[str] | None = None) -> int:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--json", action="store_true", help="machine-readable report")
    common.add_argument("--out", default="out", help="output folder (default: ./out)")
    parser = argparse.ArgumentParser(prog="python -m fx", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, fn, helptext in [
        ("validate", cmd_validate, "schema and reference checks only"),
        ("render", cmd_render, "render frames, buffers and review sheets"),
        ("audit", cmd_audit, "audit the last render"),
        ("build", cmd_build, "validate, render, audit, then game sheets; exit 1 on any fault"),
        ("gif", cmd_gif, "review GIF from the last render"),
    ]:
        p = sub.add_parser(name, parents=[common], help=helptext)
        p.add_argument("spec")
        p.set_defaults(fn=fn)
        if name in ("render", "build"):
            p.add_argument("--supersample", type=_positive, default=4)
        if name == "gif":
            p.add_argument("--zoom", type=_positive, default=4)
    p = sub.add_parser("palette-check", parents=[common], help="validate a palette file")
    p.add_argument("palette")
    p.set_defaults(fn=cmd_palette_check)
    p = sub.add_parser("doctor", parents=[common], help="check Python, pinned dependencies and --out")
    p.set_defaults(fn=cmd_doctor)
    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
