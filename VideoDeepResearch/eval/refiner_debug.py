import json
import re
from pathlib import Path
from typing import Any, Optional, Union


def sanitize_path_component(name: str) -> str:
    if not name or not str(name).strip():
        return "unknown"
    s = str(name).strip()
    s = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", s)
    s = s.strip(". ") or "unknown"
    if len(s) > 200:
        s = s[:200]
    return s


def ensure_outputs_dir(component_dir: Union[str, Path]) -> Path:
    p = Path(component_dir)
    out = p / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_json(output_dir: Union[str, Path], filename: str, obj: Any) -> None:
    p = Path(output_dir) / filename
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_text(output_dir: Union[str, Path], filename: str, content: str) -> None:
    p = Path(output_dir) / filename
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write(content if content is not None else "")


def resolve_debug_root(explicit: Optional[str] = None) -> Optional[str]:
    import os

    r = (explicit or os.environ.get("REFINER_DEBUG_ROOT") or "").strip()
    return r or None
