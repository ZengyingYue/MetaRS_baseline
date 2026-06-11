import pickle, pprint, sys

class StubModule:
    def __init__(self, name): self.__name__ = name
    def __call__(self, *a, **k):
        return {"__stub__": self.__name__, "args": a, "kwargs": k}

class Stub:
    def __init__(self, *a, **k):
        self._a, self._k = a, k
    def __repr__(self):
        return f"<Stub {type(self).__qualname__} args={self._a} kwargs={self._k}>"

class StubUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except Exception:
            t = type(name, (Stub,), {"__module__": module, "__qualname__": f"{module}.{name}"})
            return t

with open("author/config.pkl", "rb") as f:
    cfg = StubUnpickler(f).load()

def san(o, depth=0):
    if depth > 15:
        return "...DEPTH..."
    if isinstance(o, dict):
        return {san(k, depth+1): san(v, depth+1) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [san(x, depth+1) for x in o]
    if isinstance(o, (str, int, float, bool, type(None))):
        return o
    # albumentations / objects: try class name + safe dict
    try:
        cn = type(o).__name__
    except Exception:
        cn = "?"
    d = getattr(o, "__dict__", None)
    if d:
        try:
            return {f"<{cn}>": san(d, depth+1)}
        except Exception:
            pass
    try:
        return f"<{cn}: {repr(o)[:200]}>"
    except Exception:
        return f"<{cn}>"

import json
print(json.dumps(san(cfg), indent=1, ensure_ascii=False, default=str))
