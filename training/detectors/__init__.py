import os
import sys
import importlib
import warnings

current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
project_root_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(project_root_dir)

from metrics.registry import DETECTOR

# If slowfast is optional, guard it too
try:
    from .utils import slowfast  # noqa: F401
except Exception as e:
    warnings.warn(f"[DeepfakeBench] Optional dependency 'slowfast' not available: {e}")

def _safe_import(module_path: str, names=None):
    """
    Import module_path and optionally pull names from it.
    If it fails for ANY reason (ImportError, RuntimeError, missing file, etc.),
    just warn and continue so other detectors can still be used.
    """
    try:
        m = importlib.import_module(module_path, package=__name__)
        if names:
            for n in names:
                getattr(m, n)
        return True
    except Exception as e:
        warnings.warn(f"[DeepfakeBench] Skipping detector {module_path}: {e}")
        return False


# --- Core/common detectors: these SHOULD work; keep as safe anyway ---
_safe_import(".facexray_detector", ["FaceXrayDetector"])
_safe_import(".xception_detector", ["XceptionDetector"])
_safe_import(".efficientnetb4_detector", ["EfficientDetector"])
_safe_import(".resnet34_detector", ["ResnetDetector"])
_safe_import(".f3net_detector", ["F3netDetector"])
_safe_import(".meso4_detector", ["Meso4Detector"])
_safe_import(".meso4Inception_detector", ["Meso4InceptionDetector"])
_safe_import(".spsl_detector", ["SpslDetector"])
_safe_import(".core_detector", ["CoreDetector"])
_safe_import(".capsule_net_detector", ["CapsuleNetDetector"])
_safe_import(".srm_detector", ["SRMDetector"])
_safe_import(".ucf_detector", ["UCFDetector"])
_safe_import(".recce_detector", ["RecceDetector"])
_safe_import(".fwa_detector", ["FWADetector"])
_safe_import(".ffd_detector", ["FFDDetector"])
_safe_import(".videomae_detector", ["VideoMAEDetector"])
_safe_import(".clip_detector", ["CLIPDetector"])
_safe_import(".timetransfromer_detector", ["TimeTransfromerDetector"])
_safe_import(".xclip_detector", ["XCLIPDetector"])
_safe_import(".sbi_detector", ["SBIDetector"])
_safe_import(".ftcn_detector", ["FTCNDetector"])
_safe_import(".i3d_detector", ["I3DDetector"])
_safe_import(".altfreezing_detector", ["AltFreezingDetector"])
_safe_import(".stil_detector", ["STILDetector"])
_safe_import(".lsda_detector", ["LSDADetector"])

# --- The problematic ones (often pull extra datasets/assets) ---
_safe_import(".sladd_detector", ["SLADDXceptionDetector"])      # pulls SLADD pipeline (often dlib predictor)
_safe_import(".pcl_xception_detector", ["PCLXceptionDetector"]) # pulls I2G / lmdb / dataset.library in your case

_safe_import(".iid_detector", ["IIDDetector"])
_safe_import(".lrl_detector", ["LRLDetector"])
_safe_import(".rfm_detector", ["RFMDetector"])
_safe_import(".uia_vit_detector", ["UIAViTDetector"])
_safe_import(".multi_attention_detector", ["MultiAttentionDetector"])
_safe_import(".sia_detector", ["SIADetector"])
