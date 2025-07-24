from .pretrain import *
from .sft import *

CONVERTORS = {
    "pretrain": PretrainConvertor,
    "sft": SftConvertor
}