# -*- mode: python ; coding: utf-8 -*-

# -*- mode: python ; coding: utf-8 -*-
# 添加以下部分代码
from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import copy_metadata

# 设置 streamlit 运行时目录,一般在 Lib/site-packages 里
datas = [("C:\Users\ljian\AppData\Roaming\Python\Python311\site-packages/streamlit/runtime","./streamlit/runtime")]
datas += collect_data_files("streamlit")
datas += copy_metadata("streamlit")


a = Analysis(
    ['Home.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=['streamlit'],
    hookspath=['hook'],
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
    a.binaries,
    a.datas,
    [],
    name='Home',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['your_icon.ico'],
)
