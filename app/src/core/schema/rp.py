r""" This module contains definitions of the schemas used in the project. These
are used to build msgspec writers and readers.

The schemas are defined as lists of tuples, where each tuple contains the name
and type of the field.

"""

from core.data_types import SignalType
from core.quality_signals.content import content_schema
from core.quality_signals.repetitions import repetitions_schema
from core.quality_signals.natural_language import natural_language_schema
from core.quality_signals.lines import lines_schema
from core.quality_signals.classifiers import classifier_schema
from core.quality_signals.importance_weights import importance_weights_schema

METADATA_SCHEMA = [
    ("cc_net_source", str),
    ("cc_segment", str),
    ("shard_id", str),
    ("url", str),
    ("source_domain", str),
    ("language", str),
    ("snapshot_id", str)
]

QUALITY_SIGNALS_SCHEMA = [
    ("ccnet_length", SignalType),
    ("ccnet_original_length", SignalType),
    ("ccnet_nlines", SignalType),
    ("ccnet_original_nlines", SignalType),
    ("ccnet_language_score", SignalType),
    ("ccnet_perplexity", SignalType),
    ("ccnet_bucket", SignalType),
    *content_schema(),
    *natural_language_schema(),
    *repetitions_schema(),
    *lines_schema(),
    *classifier_schema(),
    *importance_weights_schema(),
]

RP_SIGNAL_SCHEMA = [
    ("id", str),
    ("id_int", int),
    ("metadata", METADATA_SCHEMA),
    ("quality_signals", QUALITY_SIGNALS_SCHEMA)
]
