import glob
import importlib
import os

DATAMODULE_REGISTRY = {}
_DATAMODULES_DISCOVERED = False


def register_datamodule(name):
    def decorator(cls):
        DATAMODULE_REGISTRY[name] = cls
        return cls
    return decorator


def ensure_datamodules_discovered():
    global _DATAMODULES_DISCOVERED
    if _DATAMODULES_DISCOVERED:
        return

    import_modules(os.path.dirname(__file__), "spurs.datamodules")
    _DATAMODULES_DISCOVERED = True


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
