# PyInstaller onedir specification. Build only in a clean native platform job.
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

datas = (
    collect_data_files("cove_sensory_mcp", includes=["assets/self_test/*"])
    + copy_metadata("mcp")
    + copy_metadata("keyring")
)
hiddenimports = collect_submodules("keyring.backends")

analysis = Analysis(
    ["src/cove_sensory_mcp/__main__.py"],
    pathex=["src"],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["pytest", "tests"],
    noarchive=False,
)
pyz = PYZ(analysis.pure)
exe = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="cove-sensory-mcp",
    console=True,
)
coll = COLLECT(
    exe,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=False,
    name="cove-sensory-mcp",
)
