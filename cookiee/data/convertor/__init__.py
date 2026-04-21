from .pretrain import *
from .midtrain import *
from .sft import *
from .preference import *

CONVERTORS = {
    "pretrain": PretrainConvertor,
    "midtrain": MidtrainConvertor, 
    "sft": SftConvertor,
    "preference_pretrain": PreferenceConvertor
}