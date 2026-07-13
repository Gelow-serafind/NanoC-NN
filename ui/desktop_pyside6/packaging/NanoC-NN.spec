# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

repo_root = Path.cwd()
entry = repo_root / "ui" / "desktop_pyside6" / "main.py"

datas = [
    (str(repo_root / "src"), "src"),
    (str(repo_root / "tdd" / "fixtures"), "tdd/fixtures"),
    (str(repo_root / "ui" / "desktop_pyside6"), "ui/desktop_pyside6"),
]

block_cipher = None

a = Analysis(
    [str(entry)],
    pathex=[str(repo_root), str(repo_root / "src")],
    binaries=[],
    datas=datas,
    hiddenimports=[
        "onnx",
        "numpy",
        "nanoc_nn",
        "nanoc_nn.cli",
        "nanoc_nn.pipeline.cli",
        "nanoc_nn.converter.cli",
        "nanoc_nn.codegen.cli",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="NanoC-NN",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="NanoC-NN",
)

app = BUNDLE(
    coll,
    name="NanoC-NN.app",
    icon=str(repo_root / "ui" / "desktop_pyside6" / "packaging" / "app_icon.icns"),
    bundle_identifier="com.nanocnn.desktop",
    info_plist={
        "CFBundleName": "NanoC-NN",
        "CFBundleDisplayName": "NanoC-NN",
        "CFBundleShortVersionString": "0.1.0",
        "CFBundleVersion": "0.1.0",
        "NSHighResolutionCapable": True,
    },
)
