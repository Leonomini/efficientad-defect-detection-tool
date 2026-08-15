import yaml
from types import SimpleNamespace

def load_config(path: str):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return dict_to_namespace(cfg)


# dict → attribute access (cfg.training.steps)
def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{
            k: dict_to_namespace(v) for k, v in d.items()
        })
    elif isinstance(d, list):
        return [dict_to_namespace(v) for v in d]
    else:
        return d


def namespace_to_dict(obj):
    if isinstance(obj, list):
        return [namespace_to_dict(x) for x in obj]
    if hasattr(obj, "__dict__"):
        return {
            key: namespace_to_dict(value)
            for key, value in vars(obj).items()
        }
    return obj