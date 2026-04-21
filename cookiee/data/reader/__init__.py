from .base import *
from .hf_reader import *
from .json_reader import *


READERS = {
    "hf": HfReader,
    "json": JsonReader,
    "jsonl": JsonlReader,
    "jsonl_generator": JsonlGeneratorReader,
}
