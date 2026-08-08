# PyInstaller onedir specification. Build only in a clean native platform job.
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

project_root = Path(SPECPATH).parent

datas = (
    collect_data_files("cove_sensory_mcp", includes=["assets/self_test/*"])
    + copy_metadata("mcp")
    + copy_metadata("keyring")
    + [
        (str(project_root / "LICENSE"), "."),
        (str(project_root / "NOTICE"), "."),
        (str(project_root / "THIRD_PARTY_NOTICES.md"), "."),
    ]
)
hiddenimports = collect_submodules("keyring.backends")

analysis = Analysis(
    [str(project_root / "src/cove_sensory_mcp/__main__.py")],
    pathex=[str(project_root / "src")],
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
