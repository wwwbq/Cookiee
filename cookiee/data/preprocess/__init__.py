from .base import *
from .pretrain import *
from .sft import *

PREPROCESSOR = {
    "pretrain": PretrainPreprocessor,
    "sft": SftPreprocessor
}