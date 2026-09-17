# -*- mode: python ; coding: utf-8 -*-
# PyInstaller build config: bundles streamlit + transformers/torch metadata and the src/ package
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

datas = []
datas += collect_data_files("streamlit")
datas += copy_metadata("streamlit")
datas += collect_data_files("transformers")
datas += copy_metadata("transformers")
datas += collect_data_files("sentence_transformers")
datas += copy_metadata("sentence_transformers")
datas += copy_metadata("tqdm")
datas += copy_metadata("regex")
datas += copy_metadata("requests")
datas += copy_metadata("packaging")
datas += copy_metadata("filelock")
datas += copy_metadata("numpy")
datas += copy_metadata("tokenizers")
datas += copy_metadata("huggingface_hub")
datas += copy_metadata("safetensors")
datas += copy_metadata("pyyaml")
datas += copy_metadata("scikit-learn")
datas += [("streamlit_app.py", "."), ("src", "src")]

hiddenimports = []
hiddenimports += collect_submodules("streamlit")
hiddenimports += collect_submodules("transformers")
hiddenimports += collect_submodules("sentence_transformers")
hiddenimports += [
    "src.extract_text",
    "src.similarity",
    "src.ai_detection_check",
    "src.teams_auth",
    "src.teams_client",
    "msal",
    "dotenv",
    "pptx",
    "striprtf",
    "striprtf.striprtf",
]

a = Analysis(
    ["run_app.py"],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="AssignmentIntegrityChecker",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
)
