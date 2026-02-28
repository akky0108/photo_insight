import sys
from pathlib import Path
import yaml

def deep_merge(a: dict, b: dict) -> dict:
    """Merge b into a (b wins)."""
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out

def load_yaml(p: Path) -> dict:
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{p} must be a mapping at top-level")
    return data

def main():
    if len(sys.argv) != 4:
        print("Usage: render_config.py <base.yaml> <env.yaml> <out.yaml>", file=sys.stderr)
        sys.exit(2)

    base_p = Path(sys.argv[1])
    env_p = Path(sys.argv[2])
    out_p = Path(sys.argv[3])

    base = load_yaml(base_p)
    env = load_yaml(env_p)

    merged = deep_merge(base, env)

    out_p.parent.mkdir(parents=True, exist_ok=True)
    with out_p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(merged, f, sort_keys=False, allow_unicode=True)

if __name__ == "__main__":
    main()
