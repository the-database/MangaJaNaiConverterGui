from spandrel import (
    MAIN_REGISTRY,
    ArchRegistry,
    ArchSupport,
)

from .architectures import FDAT

CUSTOM_REGISTRY = ArchRegistry()

CUSTOM_REGISTRY.add(
    ArchSupport.from_architecture(FDAT.FDATArch()),
)

def install(*, ignore_duplicates: bool = False) -> list[ArchSupport]:
    """
    Try to install the extra architectures into the main registry.

    If `ignore_duplicates` is True, the function will not raise an error
    if the installation fails due to any of the architectures having already
    been installed (but they won't be replaced by ones from this package).
    """
    return MAIN_REGISTRY.add(*CUSTOM_REGISTRY, ignore_duplicates=ignore_duplicates)


__all__ = [
    "CUSTOM_REGISTRY",
    "install",
]