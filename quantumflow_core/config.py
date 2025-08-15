import yaml, os

def load_config(path: str | None = None):
    if path is None:
        path = os.environ.get("QF_CONFIG", "configs/dev.yaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
