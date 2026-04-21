from .base import *
from .pretrain import *
from .sft import *
from .preference import *

PREPROCESSOR = {
    "pretrain": PretrainPreprocessor,
    "midtrain": PretrainPreprocessor,
    "sft": SftPreprocessor,
    "preference_pretrain": PreferencePretrainPreprocessor
}