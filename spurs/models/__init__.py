import glob
import importlib
import os

MODEL_REGISTRY = {}
_MODELS_DISCOVERED = False


def register_model(name):
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def ensure_models_discovered():
    global _MODELS_DISCOVERED
    if _MODELS_DISCOVERED:
        return

    import_modules(os.path.dirname(__file__), "spurs.models", excludes=['protein_structure_prediction'])
    _MODELS_DISCOVERED = True


def import_modules(models_dir, namespace, excludes=None):
    excludes = excludes or []
    for path in glob.glob(models_dir + '/**', recursive=True)[1:]:
        if any(e in path for e in excludes):
            continue

        file = os.path.split(path)[1]
        if (
            not file.startswith("_")
            and not file.startswith(".")
            and (file.endswith(".py") or os.path.isdir(path))
        ):
            module_name = file[: file.find(".py")] if file.endswith(".py") else file
            _namespace = path.replace('/', '.')
            _namespace = _namespace[_namespace.find(namespace): _namespace.rfind('.' + module_name)]
            importlib.import_module(_namespace + "." + module_name)

