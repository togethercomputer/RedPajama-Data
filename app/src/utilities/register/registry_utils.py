import inspect
from typing import Tuple, List, Type

from core.quality_signals.base import RPSBase

__all__ = [
    "get_callables_from_module", "signal_schema"
]

_SIG_PREF = RPSBase.RPS_PREFIX


def get_callables_from_module(module: object) -> List[Type[RPSBase]]:
    r""" Returns a list of signal class references that are defined in the
    module.

    Args:
        module: The module to search for signal classes, obtained via
            `sys.modules[__name__]`.

    Returns:
        A list of signal class references.
    """

    def _sig_func_predicate(mem: object):
        return inspect.isclass(mem) and mem.__name__.startswith(_SIG_PREF)

    return [cls for _, cls in inspect.getmembers(module, _sig_func_predicate)]


def signal_schema(module: object) -> List[Tuple[str, Type]]:
    r""" Returns a list of signal names and their data types, defining the
    schema for signals. """
    return list(map(
        lambda cls: (cls.__name__.lower(), cls.DATA_TYPE),
        get_callables_from_module(module=module)
    ))
