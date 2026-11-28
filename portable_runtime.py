"""Checkout-relative bootstrap for the standalone Newton scene tools."""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parent

def bootstrap():
    install_path_audit()
    package = ROOT / "outputs/ax50_visualskin_v3_0_0"
    if not (package / "ax50_visualskin/__init__.py").is_file():
        raise RuntimeError("AX50 assets are missing. Run: python scripts/bootstrap_scenes.py")
    for folder in (ROOT / "scripts/nov20", package / "tools", package):
        if str(folder) not in sys.path:
            sys.path.insert(0, str(folder))
    os.environ.setdefault("MUJOCO_GL", "egl")
    (ROOT / "outputs/nov20").mkdir(parents=True, exist_ok=True)
    import shutil
    if shutil.which("ffmpeg") is None:
        import imageio_ffmpeg
        tools = ROOT / "outputs/portable_tools"
        tools.mkdir(parents=True, exist_ok=True)
        exe = tools / "ffmpeg"
        if not exe.exists():
            exe.symlink_to(imageio_ffmpeg.get_ffmpeg_exe())
        os.environ["PATH"] = str(tools) + os.pathsep + os.environ.get("PATH", "")
    import ax50_visualskin
    ax50_visualskin.activate_newton()


def parse_scene_xml(path, *args, **kwargs):
    """Resolve referenced assets before a scene subtree is moved to a new XML file."""
    import xml.etree.ElementTree as ET
    tree = ET.parse(path, *args, **kwargs)
    if not isinstance(path, (str, Path)):
        return tree
    base = Path(path).resolve().parent
    for node in tree.iter():
        for attr in ("file", "filename"):
            value = node.get(attr)
            if value and not Path(value).is_absolute() and not value.startswith("package://"):
                node.set(attr, str((base / value).resolve()))
    return tree


_audit_installed = False

def install_path_audit():
    """Optional test guard: reject file reads under explicitly forbidden old checkouts."""
    global _audit_installed
    if _audit_installed:
        return
    import json
    prefixes = [str(Path(p).resolve()) for p in json.loads(os.environ.get("EVOLUTION_FORBID_ROOTS", "[]"))]
    if not prefixes:
        return
    def audit(event, args):
        if event != "open" or not args or not isinstance(args[0], (str, bytes)):
            return
        path = os.path.realpath(os.fsdecode(args[0]))
        if any(path == p or path.startswith(p + os.sep) for p in prefixes):
            raise PermissionError("Nonportable dependency read blocked: " + path)
    sys.addaudithook(audit)
    _audit_installed = True
