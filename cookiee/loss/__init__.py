from .base import *
from .domain_loss import *
from .metric_aggregator import *
from .focal_loss import *
from .preference_loss import *


LOSSES = {
    "domain_loss": DomainLoss,
    "focal_loss": FocalLoss,
    "preference_loss": PreferenceLoss,
}
