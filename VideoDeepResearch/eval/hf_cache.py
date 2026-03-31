import os

# Default scratch layout (override with VDR_RUNTIME_ROOT or NEXUS_RUNTIME_ROOT).
NEXUS_RUNTIME_ROOT_DEFAULT = "/share/data/drive_1/.cache"
NEXUS_HF_HOME = os.path.join(NEXUS_RUNTIME_ROOT_DEFAULT, "huggingface")
SHARED_HF_HOME_DEFAULT = "/share/data/drive_1/huggingface_cache"


def _runtime_root() -> str:
    return (
        os.environ.get("VDR_RUNTIME_ROOT")
        or os.environ.get("NEXUS_RUNTIME_ROOT")
        or NEXUS_RUNTIME_ROOT_DEFAULT
    )


def ensure_hf_cache_env():
    rr = _runtime_root()
    os.makedirs(rr, exist_ok=True)

    default_hf = os.path.join(rr, "huggingface")
    shared_hf = os.environ.get("VDR_SHARED_HF_HOME", SHARED_HF_HOME_DEFAULT)
    if "HF_HOME" in os.environ:
        resolved_hf = os.environ["HF_HOME"]
    elif shared_hf and os.path.isdir(shared_hf):
        resolved_hf = shared_hf
    else:
        resolved_hf = default_hf
    root = os.environ.setdefault("HF_HOME", resolved_hf)
    os.makedirs(root, exist_ok=True)
    hub = os.path.join(root, "hub")
    os.makedirs(hub, exist_ok=True)
    os.environ.setdefault("HF_HUB_CACHE", hub)

    paddle = os.path.join(rr, "paddlex")
    os.environ.setdefault("PADDLE_PDX_CACHE_HOME", paddle)
    os.makedirs(paddle, exist_ok=True)

    triton = os.path.join(rr, "triton")
    os.environ.setdefault("TRITON_CACHE_DIR", triton)
    os.makedirs(triton, exist_ok=True)

    tmp = os.path.join(rr, "tmp")
    os.environ.setdefault("TMPDIR", tmp)
    os.makedirs(tmp, exist_ok=True)

    torch_home = os.path.join(rr, "torch")
    os.environ.setdefault("TORCH_HOME", torch_home)
    os.makedirs(torch_home, exist_ok=True)
