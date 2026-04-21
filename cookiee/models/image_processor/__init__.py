from helper import Registry

IMAGE_PROCESSOR = Registry()

from .llava_image_processor import *


def build_image_processor(name, use_fast=True, **kwargs):
    image_processor_type = IMAGE_PROCESSOR.build(name)
    if use_fast:
        if "fast" not in image_processor_type:
            image_processor_cls = image_processor_type["type"]
            raise Warning(f"image_processor: {name} don`t support fast image processor, use {image_processor_cls.__name__} instead. ")
        else:
            image_processor_cls = image_processor_type["fast"]
    else:
        image_processor_cls = image_processor_type["type"]
    
    image_process_kwargs = image_processor_type["default_kwargs"]
    image_process_kwargs.update(**kwargs)

    return image_processor_cls, image_process_kwargs
