import math
import torch
from PIL import Image
from PIL.Image import Image as ImageObject
from io import BytesIO
from copy import deepcopy
from abc import abstractmethod

from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, TypedDict, Union

from transformers import PreTrainedTokenizer, ProcessorMixin
from transformers.image_processing_utils import BaseImageProcessor



class EncodedImage(TypedDict):
        path: Optional[str]
        bytes: Optional[bytes]


ImageInput = Union[str, EncodedImage, ImageObject]
VideoInput = str


class BasePlugin:
    def __init__(
        self, 
        processor: ProcessorMixin, 
        image_token: Optional[str], 
        video_token: Optional[str] = None,
        image_placeholder: Optional[str] = "<image>",
        video_placeholder: Optional[str] = "<video>",
    ) -> None:
        self.processor = processor

        self.image_token = image_token if image_token is not None else getattr(processor, "image_token", None)
        self.video_token = video_token if video_token is not None else getattr(processor, "video_token", None)

        self.image_placeholder = image_placeholder
        self.video_placeholder = video_placeholder


    def _validate_input(
        self,
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
    ) -> None:
        r"""
        Validates if this model accepts the input modalities.
        """
        if images is not None and len(images) != 0 and self.image_token is None:
            raise ValueError("This model does not support image input.")

        if videos is not None and len(videos) != 0 and self.video_token is None:
            raise ValueError("This model does not support video input.")


    def _preprocess_image(self, image: "ImageObject", **kwargs) -> "ImageObject":
        r"""
        Pre-processes a single image.
        """
        image_resolution: int = kwargs.get("image_resolution")
        if max(image.width, image.height) > image_resolution:
            resize_factor = image_resolution / max(image.width, image.height)
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height), resample=Image.NEAREST)

        if image.mode != "RGB":
            image = image.convert("RGB")

        return image


    def _regularize_images(self, images: Sequence["ImageInput"], **kwargs) -> List["ImageObject"]:
        r"""
        Regularizes images to avoid error. Including reading and pre-processing.
        """
        results = []
        for image in images:
            if isinstance(image, str):
                image = Image.open(image)
            elif isinstance(image, dict):
                if image["bytes"] is not None:
                    image = Image.open(BytesIO(image["bytes"]))
                else:
                    image = Image.open(image["path"])

            if not isinstance(image, ImageObject):
                raise ValueError("Expect input is a list of Images, but got {}.".format(type(image)))

            results.append(self._preprocess_image(image, **kwargs))

        return results


    def align_messages(self, messages: Optional[Union[str, List[Dict]]]):
        r"""
            将非对话形式的数据转换成对话形式的格式，并返回原始数据的type
        """

        messages_type = type(messages)

        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        return messages, messages_type
        

    @abstractmethod
    def preprocess_multi_modal_messages(
        self,
        messages: Sequence[Dict[str, str]],
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        processor: Optional["ProcessorMixin"] = None,
    ) -> List[Dict[str, str]]:
        r"""
        Pre-processes input messages before tokenization for VLMs.
        """
        pass
        

    @abstractmethod
    def collator_fn(
        self,
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        processor: "ProcessorMixin",
    ):
        pass


    def create_visual_mask(self, labels: torch.Tensor, tokenizer: PreTrainedTokenizer) -> torch.Tensor:
        # Returns a mask for visual tokens in the labels when multimodal pretraining.
        raise NotImplementedError


class LlavaPlugin(BasePlugin):
    def preprocess_multi_modal_messages(
        self, 
        messages: Sequence[Dict[str, str]],
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"] = [],
        processor: Optional["ProcessorMixin"] = None,
    ):
        if processor is None:
            processor = self.processor
        
        self._validate_input(images, videos)

        num_image_tokens = 0
        image_seqlen = getattr(processor, "image_seqlen")

        messages, messages_type = self.align_messages(deepcopy(messages))

        for message in messages:
            content = message["content"]
            num_image_tokens += content.count(self.image_placeholder)
            message["content"] = content.replace(self.image_placeholder, self.image_token * image_seqlen)

        if len(images) != num_image_tokens:
            raise ValueError("The number of images does not match the number of {} tokens".format(self.image_placeholder))

        return messages if messages_type != str else messages[0]["content"]


    def collator_fn(
        self,
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"] = [],
        processor: "ProcessorMixin" = None,
    ):
        r"""
        Returns:
            pixel_values: tensor with shape (B, C, H, W)

        """
        self._validate_input(images, videos)

        if processor is None:
            processor = self.processor

        image_processor: "BaseImageProcessor" = getattr(processor, "image_processor")
        video_processor: "BaseImageProcessor" = getattr(processor, "video_processor", image_processor)

        input_dict = {"images": None}  # default key

        if len(images) != 0:
            images = self._regularize_images(
                images,
                image_resolution=getattr(processor, "image_resolution", 512),
            )
            input_dict["images"] = images

        mm_inputs = {}
        if image_processor != video_processor:
            if input_dict.get("images") is not None:
                mm_inputs.update(image_processor(input_dict["images"], return_tensors="pt"))
            if input_dict.get("videos") is not None:
                mm_inputs.update(video_processor(input_dict["videos"], return_tensors="pt"))
        elif input_dict.get("images") is not None or input_dict.get("videos") is not None:  # same processor (qwen2-vl)
            mm_inputs.update(image_processor(**input_dict, return_tensors="pt"))

        return mm_inputs


    def create_visual_mask(self, labels: torch.Tensor, tokenizer: PreTrainedTokenizer) -> torch.Tensor:
        image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        mask = (labels == image_token_id)
        return mask


class Qwen2vlPlugin(BasePlugin):
    def _preprocess_image(self, image: "ImageObject", **kwargs) -> "ImageObject":
        image = super()._preprocess_image(image, **kwargs)
        if min(image.width, image.height) < 28:
            width, height = max(image.width, 28), max(image.height, 28)
            image = image.resize((width, height), resample=Image.NEAREST)

        if image.width / image.height > 200:
            width, height = image.height * 180, image.height
            image = image.resize((width, height), resample=Image.NEAREST)

        if image.height / image.width > 200:
            width, height = image.width, image.width * 180
            image = image.resize((width, height), resample=Image.NEAREST)

        return image


    def preprocess_multi_modal_messages(
        self,
        messages: Sequence[Dict[str, str]],
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"] = [],
        processor: Optional["ProcessorMixin"] = None,
    ) -> List[Dict[str, str]]:
        
        if processor is None:
            processor = self.processor
        
        self._validate_input(images, videos)

        image_processor: "BaseImageProcessor" = getattr(processor, "image_processor")

        merge_length: int = getattr(image_processor, "merge_size") ** 2

        mm_inputs = self.collator_fn(images, videos, processor)

        image_grid_thw = mm_inputs.get("image_grid_thw", [])
        video_grid_thw = mm_inputs.get("video_grid_thw", [])

        num_image_tokens, num_video_tokens = 0, 0

        messages, messages_type = self.align_messages(deepcopy(messages))

        for message in messages:
            content = message["content"]
            while self.image_placeholder in content:
                if num_image_tokens >= len(image_grid_thw):
                    raise ValueError("`len(images)` is less than the number of {} tokens.".format(self.image_placeholder))

                content = content.replace(
                    self.image_placeholder,
                    "<|vision_start|>{}<|vision_end|>".format(
                        self.image_token * (image_grid_thw[num_image_tokens].prod() // merge_length)
                    ),
                    1,
                )
                num_image_tokens += 1

            while self.video_placeholder in content:
                if num_video_tokens >= len(video_grid_thw):
                    raise ValueError("`len(videos)` is less than the number of {} tokens.".format(self.video_placeholder))

                content = content.replace(
                    self.video_placeholder,
                    "<|vision_start|>{}<|vision_end|>".format(
                        self.video_token * (video_grid_thw[num_video_tokens].prod() // merge_length)
                    ),
                    1,
                )
                num_video_tokens += 1

            message["content"] = content

        if len(images) != num_image_tokens:
            raise ValueError("The number of images does not match the number of {} tokens".format(self.image_placeholder))

        if len(videos) != num_video_tokens:
            raise ValueError("The number of videos does not match the number of {} tokens".format(self.video_placeholder))

        return messages if messages_type != str else messages[0]["content"]


    def collator_fn(
        self,
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"] = [],
        processor: "ProcessorMixin" = None,
    ):
        r"""
        Returns: (qwen2-vl)
            pixel_values: tensor with shape (num_patches, patch_dim)
            image_grid_thw: tensor with shape (num_images, 3), where the three numbers are time, width, height

        It holds num_patches == torch.prod(image_grid_thw)

        """
        self._validate_input(images, videos)

        if processor is None:
            processor = self.processor

        image_processor: "BaseImageProcessor" = getattr(processor, "image_processor")
        video_processor: "BaseImageProcessor" = getattr(processor, "video_processor", image_processor)

        input_dict = {"images": None}  # default key

        if images is not None and len(images) != 0:
            images = self._regularize_images(
                images,
                image_resolution=getattr(processor, "image_resolution", 512),
            )
            input_dict["images"] = images

        mm_inputs = {}
        if image_processor != video_processor:
            if input_dict.get("images") is not None:
                mm_inputs.update(image_processor(input_dict["images"], return_tensors="pt"))
            if input_dict.get("videos") is not None:
                mm_inputs.update(video_processor(input_dict["videos"], return_tensors="pt"))
        elif input_dict.get("images") is not None or input_dict.get("videos") is not None:  # same processor (qwen2-vl)
            mm_inputs.update(image_processor(**input_dict, return_tensors="pt"))

        return mm_inputs


    def create_visual_mask(self, labels: torch.Tensor, tokenizer: PreTrainedTokenizer) -> torch.Tensor:
        # 在Qwen2-vl-base中，每个视觉片段由<|vision_start|>和<|vision_end|>标记包围
        vision_start_token_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
        vision_end_token_id = tokenizer.convert_tokens_to_ids("<|vision_end|>")
        mask = torch.zeros_like(labels, dtype=torch.bool)

        # 对批次中的每个样本进行处理
        for i in range(labels.size(0)):
            # 找出当前样本的起始和结束位置
            start_positions = (labels[i] == vision_start_token_id).nonzero(as_tuple=True)[0]
            end_positions = (labels[i] == vision_end_token_id).nonzero(as_tuple=True)[0]

            # 处理每个视觉片段
            for start, end in zip(start_positions, end_positions):
                if end > start:
                    mask[i, start:end+1] = True

        return mask