from transformers.models.llava.image_processing_llava import LlavaImageProcessor
from transformers.models.llava.image_processing_llava_fast import LlavaImageProcessorFast

from . import IMAGE_PROCESSOR


default_image_processor_kwargs = {
    "do_pad": False,
    "crop_size": {
        "height": 336,
        "width": 336
    },
    "do_center_crop": True,
    "do_convert_rgb": True,
    "do_normalize": True,
    "do_rescale": True,
    "do_resize": True,
    "image_mean": [
        0.48145466,
        0.4578275,
        0.40821073
    ],
    "image_std": [
        0.26862954,
        0.26130258,
        0.27577711
    ],
    "resample": 3,
    "rescale_factor": 0.00392156862745098,
    "size": {
        "shortest_edge": 336
    }
}


IMAGE_PROCESSOR._do_register(
    name="llava",
    obj={
        "type": LlavaImageProcessor,
        "fast": LlavaImageProcessorFast,
        "default_kwargs": default_image_processor_kwargs
    }
)