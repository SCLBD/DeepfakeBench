# training/dataset/__init__.py
import os
import sys
import importlib
import warnings

current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
project_root_dir = os.path.dirname(parent_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
if project_root_dir not in sys.path:
    sys.path.append(project_root_dir)

def _safe_import(rel_module: str, names):
    """
    rel_module: like '.I2G_dataset'
    names: list of symbols to import from that module
    """
    try:
        m = importlib.import_module(rel_module, package=__name__)
        exported = {}
        for n in names:
            exported[n] = getattr(m, n)
        globals().update(exported)
        return True
    except Exception as e:
        warnings.warn(f"[DeepfakeBench] Skipping dataset {rel_module}: {e}")
        return False

# Core base first (if this fails, your env is fundamentally broken)
_safe_import(".abstract_dataset", ["DeepfakeAbstractBaseDataset"])

# Optional datasets (won't crash whole project if missing deps / files)
_safe_import(".I2G_dataset", ["I2GDataset"])
_safe_import(".iid_dataset", ["IIDDataset"])
_safe_import(".ff_blend", ["FFBlendDataset"])
_safe_import(".fwa_blend", ["FWABlendDataset"])
_safe_import(".lrl_dataset", ["LRLDataset"])
_safe_import(".pair_dataset", ["pairDataset"])
_safe_import(".sbi_dataset", ["SBIDataset"])
_safe_import(".lsda_dataset", ["LSDADataset"])
