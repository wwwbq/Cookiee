import os
from enum import Enum, unique
from dataclasses import dataclass
from typing import Dict, Any, Optional, Literal

@unique
class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    FUNCTION = "function"
    OBSERVATION = "observation"


@dataclass
class DatasetAttr:
    # basic configs
    dataset_name: str
    load_from: Literal["hf_hub", "ms_hub", "script", "file"] = "file"
    file_name: Optional[str] = None
    file_root: Optional[str] = None
    image_folder: Optional[str] = None
    dataset_dir: Optional[str] = None
    formatting: Literal["pretrain", "alpaca", "sharegpt"] = "alpaca"
    
    channel: Optional[str] = None # 领域数据类型，例如 math, code
    image_token: Optional[str] = None
    image_placeholder: Optional[str] = "<image>"
    

    # extra configs
    subset: Optional[str] = None
    split: str = "train"
    folder: Optional[str] = None
    num_samples: Optional[int] = None

    # common columns
    system: Optional[str] = None
    tools: Optional[str] = None
    images: Optional[str] = None
    videos: Optional[str] = None

    # pretrain columns
    text: Optional[str] = "text"

    # rlhf columns
    chosen: Optional[str] = None
    rejected: Optional[str] = None
    kto_tag: Optional[str] = None

    # alpaca columns
    prompt: Optional[str] = "instruction"
    query: Optional[str] = "input"
    response: Optional[str] = "output"
    history: Optional[str] = None

    # sharegpt columns
    messages: Optional[str] = "conversations"

    # sharegpt tags
    role_tag: Optional[str] = "from"
    content_tag: Optional[str] = "value"
    user_tag: Optional[str] = "human"
    assistant_tag: Optional[str] = "gpt"
    observation_tag: Optional[str] = "observation"
    function_tag: Optional[str] = "function_call"
    system_tag: Optional[str] = "system"

    def __repr__(self) -> str:
        return self.dataset_name


    def set_attr(self, key: str, obj: Dict[str, Any], default: Optional[Any] = None) -> None:
        setattr(self, key, obj.get(key, default))


    @classmethod
    def from_dataset_info(cls, dataset_name, dataset_info: Dict[str, Any]):
        key_info = dataset_info.pop("key")
        dataset_info.update(key_info)
        dataset_info["dataset_name"] = dataset_name
        return cls.from_dict(dataset_info)


    @classmethod
    def from_dict(cls, attr: Dict[str, Any]):
        return cls(**attr)
    

def convert_images(example, dataset_attr: DatasetAttr):
    if dataset_attr.images is None:
        return None

    images = example[dataset_attr.images]

    if images is None or len(images) == 0:
        return None
    
    images = images[:]
    if not isinstance(images, list):
        images = [images]
    
    for i in range(len(images)):
        if isinstance(images[i], str):
            image_folder = dataset_attr.image_folder if dataset_attr.image_folder else dataset_attr.dataset_dir
            images[i] = os.path.join(image_folder, images[i])

    return images


def convert_channel(example, dataset_attr: DatasetAttr):
    # 优先级：整个数据集的统一channel > 单条数据的channel
    if dataset_attr.channel:
        return dataset_attr.channel
    else:
        return example.get("channel", None)