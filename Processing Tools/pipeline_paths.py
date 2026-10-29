"""Resolve input/output case paths for pipeline batch scripts."""
import os

# Layout under repo mounts (see .devcontainer/devcontainer.json):
#   processed_data/Windtunnel_PTV_TNTI/X300cm_PTV_{above,below,center}_TNTI/{case}/{run}/
#     → Centers_*, rays.h5
#   data/Windtunnel_PTV_TNTI/X300cm_PTV_{above,below,center}_TNTI/{case}/{run}/
#     → rays_out_cpp.h5, tracks.h5part

WINDTUNNEL = "Windtunnel_PTV_TNTI"
PTV_ABOVE = "X300cm_PTV_above_TNTI"


def default_processed_base() -> str:
    return os.path.join("processed_data", WINDTUNNEL, PTV_ABOVE)


def default_data_base() -> str:
    return os.path.join("data", WINDTUNNEL, PTV_ABOVE)


def resolve_case_paths(case, *, input_base=None, output_base=None, default_output_base=None):
    """Return (input_case_dir, output_case_dir) as container-relative paths."""
    if output_base:
        out_base = os.path.join(output_base, case)
    elif default_output_base:
        out_base = os.path.join(default_output_base, case)
    else:
        out_base = os.path.join(default_data_base(), case)

    if input_base:
        in_base = os.path.join(input_base, case)
    else:
        in_base = out_base
    return in_base, out_base


def ensure_output_run_dir(output_run_path: str) -> None:
    os.makedirs(output_run_path, exist_ok=True)
