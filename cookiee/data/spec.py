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
class DatasetSpec:
    # basic configs
    dataset_name: str

    # reader type
    reader: Optional[str] = None

    # 文件名，也可以是完整的目录
    file: Optional[str] = None

    # 数据集所在目录，如果不为空，完整的数据集路径为 folder + file
    folder: Optional[str] = None

    # 图片所在目录
    image_folder: Optional[str] = None
    
    # 数据集类型
    format: Literal["content", "alpaca", "sharegpt"] = "content"
    
    # 领域数据类型，例如 math, code
    domain: Optional[str] = None
    
    # hf configs
    subset: Optional[str] = None
    data_files: Optional[Dict[str, str]] = None
    split: str = "train"
    num_samples: Optional[int] = None

    # common columns
    system: Optional[str] = None
    tools: Optional[str] = None
    images: Optional[str] = None
    videos: Optional[str] = None

    # pretrain columns
    text: Optional[str] = "text"

    # midtrain columns
    midtrain_template: Literal["sft", "plain", "none"] = "none"

    # rlhf columns
    chosen: Optional[str] = None
    rejected: Optional[str] = None

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
