import os

NEXUS_HF_HOME = "/fs/nexus-scratch/gnanesh/.cache/huggingface"


def ensure_hf_cache_env():
    root = os.environ.setdefault("HF_HOME", NEXUS_HF_HOME)
    os.makedirs(root, exist_ok=True)
    hub = os.path.join(root, "hub")
    os.makedirs(hub, exist_ok=True)
    os.environ.setdefault("HF_HUB_CACHE", hub)
