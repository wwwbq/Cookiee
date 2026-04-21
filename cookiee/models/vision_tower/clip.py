from transformers.models.clip import CLIPVisionModel

from . import VISION_TOWER


VISION_TOWER._do_register("clip", CLIPVisionModel)

