from dataclasses import dataclass
from msgspec import Struct

from typing import List, Tuple, Optional, Dict
from typing_extensions import TypeAlias

ScoreType: TypeAlias = Tuple[int, int, Optional[float]]
SignalType: TypeAlias = List[ScoreType]


@dataclass
class TextSlice:
    text: str
    start: int
    end: int

    def __len__(self):
        return len(self.text)


class InputSpec(Struct):
    raw_content: str
    url: str
    nlines: int
    original_nlines: int
    source_domain: str
    length: int
    original_length: int
    language: str
    language_score: float
    perplexity: float
    bucket: str
    digest: str
    cc_segment: str
    date_download: str


class OutputSpec(Struct):
    id: str
    id_int: int
    metadata: Dict[str, str]
    quality_signals: Dict[str, List[Tuple[int, int, Optional[float]]]]



