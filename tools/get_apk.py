#!/usr/bin/env python3
"""Fetch the published Prism app (XAPK/APK) into ./apk/.

Order: an APK/XAPK committed to apk/ on this branch wins; otherwise download the
current Play build mirrored by APKPure with EFF's apkeep. Writes apk/info.json.
"""
import json
import os
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

PKG = "com.nahid002345.prismunitconverter"
APK_DIR = Path("apk")
APK_DIR.mkdir(exist_ok=True)


def sh(cmd):
    print("+", cmd, flush=True)
    return subprocess.run(cmd, shell=True, text=True, capture_output=True)


def have_package():
    return any(APK_DIR.glob("*.xapk")) or any(APK_DIR.glob("*.apk")) or any(APK_DIR.glob("*.apkm"))


info = {"package": PKG, "source": None}
if have_package():
    info["source"] = "committed file in apk/"
else:
    # apkeep release binary (x86_64 linux)
    url = "https://github.com/EFForg/apkeep/releases/latest/download/apkeep-x86_64-unknown-linux-gnu"
    urllib.request.urlretrieve(url, "/tmp/apkeep")
    os.chmod("/tmp/apkeep", 0o755)
    r = sh(f"/tmp/apkeep -a {PKG} -d apk-pure {APK_DIR}")
    print(r.stdout, r.stderr)
    info["source"] = "APKPure mirror via apkeep"

pkgs = sorted(list(APK_DIR.glob("*.xapk")) + list(APK_DIR.glob("*.apk")) + list(APK_DIR.glob("*.apkm")))
if not pkgs:
    info["error"] = "no package downloaded"
    Path("apk/info.json").write_text(json.dumps(info, indent=2))
    print("No package downloaded", file=sys.stderr)
    sys.exit(0)  # let the bridge start anyway and report the problem

src = pkgs[0]
info["file"] = src.name
info["size_mb"] = round(src.stat().st_size / 1e6, 1)
splits = APK_DIR / "splits"
splits.mkdir(exist_ok=True)
if src.suffix in (".xapk", ".apkm") or zipfile.is_zipfile(src) and src.suffix != ".apk":
    with zipfile.ZipFile(src) as z:
        names = z.namelist()
        info["contents"] = names
        for n in names:
            if n.endswith(".apk"):
                z.extract(n, splits)
        if "manifest.json" in names:
            info["manifest"] = json.loads(z.read("manifest.json"))
else:
    (splits / src.name).write_bytes(src.read_bytes())

info["apks"] = sorted(str(p.relative_to(splits)) for p in splits.rglob("*.apk"))
Path("apk/info.json").write_text(json.dumps(info, indent=2, default=str))
print(json.dumps(info, indent=2, default=str))
