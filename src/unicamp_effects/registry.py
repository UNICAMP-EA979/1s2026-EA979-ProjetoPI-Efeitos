from types import MappingProxyType
from typing import Callable

from .image_effect import ImageEffect

_effect_registry: dict[str, ImageEffect] = {}


def register(prefix: str | None = None) -> Callable[[ImageEffect], ImageEffect]:

    def register_inner(func: ImageEffect) -> ImageEffect:
        if prefix is None:
            name = func.__qualname__
        else:
            name = f"{prefix}_{func.__qualname__}"

        if name in _effect_registry:
            raise RuntimeError(f"There is already an effect named {name}")

        _effect_registry[name] = func

        return func

    return register_inner


def get_all_effects() -> MappingProxyType[str, ImageEffect]:
    return MappingProxyType(_effect_registry)


def get_effect(name: str) -> ImageEffect:
    return _effect_registry[name]
