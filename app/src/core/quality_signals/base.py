from core.document import Document
from core.data_types import SignalType


class RPSBase:
    r""" Base class for RP signal functions. Each child class must implement
    the __call__ method. The __call__ method takes a document as input and
    returns a score. """
    DATA_TYPE = SignalType

    RPS_PREFIX: str = "RPS_"

    __slots__ = ["__field_name"]

    def __init__(self, *args, **kwargs):  # noqa
        # make sure all classes start with RPS_; this is to ensure that
        # the get_rule_based_signals function works correctly when new signal
        # functions are added
        assert self.__class__.__name__.startswith(self.RPS_PREFIX), \
            f"Name of signal function must" \
            f" start with {self.RPS_PREFIX}; got {self.__class__.__name__}"

        self.__field_name = self.__class__.__name__.lower()

    def __call__(self, document: Document):
        raise NotImplementedError

    @property
    def field_name(self):
        return self.__field_name
