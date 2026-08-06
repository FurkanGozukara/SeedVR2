"""Early runtime compatibility fixes for optional PyTorch ecosystem packages."""

from __future__ import annotations

import enum
import logging
from typing import Any


_DIFFUSERS_TORCHAO_LOGGER = "diffusers.quantizers.torchao.torchao_quantizer"
_NATIVE_ENUM_COMPAT_MARKER = "_torchao_native_enum_pytree_compat"
_WARNING_FILTER_MARKER = "_torchao_checkpoint_warning_filtered"
_DIFFUSERS_TORCHAO_CHECKPOINT_WARNING = (
    "Unable to import `torchao` Tensor objects. This may affect loading "
    "checkpoints serialized with `torchao`"
)


def ensure_native_enum_pytree_compat() -> bool:
    """Keep older TorchAO releases from registering natively opaque Enums.

    PyTorch 2.13 represents Enum subclasses as opaque values in ``torch.compile``.
    TorchAO 0.18 still calls ``register_constant`` for those classes, which is
    deprecated now and is scheduled to become an error. Skip only that obsolete
    registration while preserving ``register_constant`` for every other type.

    Returns ``True`` when native Enum handling is present and the compatibility
    wrapper is installed (or was already installed).
    """
    try:
        from torch._library.opaque_object import is_opaque_type
        from torch.utils import _pytree
    except (AttributeError, ImportError, ModuleNotFoundError):
        return False

    register_constant = _pytree.register_constant
    if getattr(register_constant, _NATIVE_ENUM_COMPAT_MARKER, False):
        return True

    class _EnumProbe(enum.Enum):
        VALUE = 1

    try:
        if not is_opaque_type(_EnumProbe):
            return False
    except (TypeError, RuntimeError):
        return False

    def register_constant_compat(cls: type[Any]) -> None:
        if isinstance(cls, type) and issubclass(cls, enum.Enum):
            try:
                if is_opaque_type(cls):
                    return None
            except (TypeError, RuntimeError):
                pass
        return register_constant(cls)

    setattr(register_constant_compat, _NATIVE_ENUM_COMPAT_MARKER, True)
    register_constant_compat._torchao_compat_original = register_constant
    _pytree.register_constant = register_constant_compat
    return True


class _IrrelevantTorchAoCheckpointWarningFilter(logging.Filter):
    """Hide a Diffusers warning for a checkpoint format SeedVR2 never loads."""

    def filter(self, record: logging.LogRecord) -> bool:
        return record.getMessage() != _DIFFUSERS_TORCHAO_CHECKPOINT_WARNING


def suppress_irrelevant_diffusers_torchao_warning() -> None:
    """Suppress only Diffusers' TorchAO-serialized-checkpoint warning.

    SeedVR2 loads ordinary SafeTensors/PyTorch state dictionaries and GGUF files
    through its own model loader. It does not ask Diffusers to deserialize TorchAO
    tensor subclasses, so this import-time warning cannot describe a supported
    SeedVR2 checkpoint path.
    """
    logger = logging.getLogger(_DIFFUSERS_TORCHAO_LOGGER)
    if getattr(logger, _WARNING_FILTER_MARKER, False):
        return
    logger.addFilter(_IrrelevantTorchAoCheckpointWarningFilter())
    setattr(logger, _WARNING_FILTER_MARKER, True)


def configure_runtime_compat() -> None:
    """Install compatibility behavior before Diffusers or TorchAO is imported."""
    ensure_native_enum_pytree_compat()
    suppress_irrelevant_diffusers_torchao_warning()
