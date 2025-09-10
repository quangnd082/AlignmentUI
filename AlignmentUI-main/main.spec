# -*- mode: python ; coding: utf-8 -*-
import sys
import os
import glob
from PyInstaller.utils.hooks import collect_data_files

sys.path.append('Libs/')
sys.path.append('Libs/cameras/')

hiddenimports = ['pypylon', 'pypylon.pylon', 'pypylon.genicam']

# Lấy toàn bộ file trong thư mục res
datas = collect_data_files('res', includes=['**/*'])

# Thêm DLL của MVS
dll_path = ('C:\\Program Files (x86)\\Common Files\\MVS\\Runtime\\Win64_x64\\MvCameraControl.dll', '.')

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[dll_path] + datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='main',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
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
    name='main',
)
