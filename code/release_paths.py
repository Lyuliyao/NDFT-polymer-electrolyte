from pathlib import Path
import os

REPO = Path(__file__).resolve().parents[1]
GROUP = str(REPO / "work" / "runtime")
E75ROOT = os.environ.get("NDFT_EPS75_ROOT", str(Path(GROUP) / "salt_in_polymer_eps75"))
E2ROOT = os.environ.get("NDFT_EPS2_ROOT", str(Path(GROUP) / "salt_in_polymer_eps2"))
EPSXROOT = os.environ.get("NDFT_EPSX_ROOT", str(Path(GROUP) / "salt_in_polymer_epsx"))
EPS_ROOTS = {e: os.environ.get(f"NDFT_EPS{e.replace('.', '')}_ROOT", str(Path(GROUP) / f"salt_in_polymer_eps{e.replace('.', '')}"))
             for e in ("7.5", "5", "4", "3", "2")}
MAIN = E75ROOT
FIGURES = REPO / "results" / "figures"
