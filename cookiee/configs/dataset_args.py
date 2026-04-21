from typing import Any, Optional, Union
from dataclasses import dataclass, field


@dataclass
class DatasetArguments:
    dataset: Optional[str] = field(
        default = None,
        metadata = {
            "help": "name of datasets, separated by comma when multiple datasets are used"
        },
    )
    eval_dataset: Optional[str] = field(
        default = None,
        metadata = {
            "help": "name of evaluation datasets, separated by comma when multiple datasets are used"
        },
    )
    dataset_dir: Optional[str] = field(
        default = "./data",
        metadata = {
            "help": "path to dataset_info.json"
        },
    )
    max_seq_length: Optional[int] = field(
        default = 512,
        metadata = {
            "help": "maximum sequence length, including prompt and response"
        },
    )
    chat_template: Optional[str] = field(
        default = None,
        metadata = {
            "help": "path to chat template"
        },
    )
    image_token: Optional[str] = field(
        default = "<image>",
        metadata = {
            "help": "token to use for image"
        },
    )
    packing: bool = field(
        default = False,
        metadata = {
            "help": "whether to pack samples into batches, default True when pretrain"
        },
    )
    use_bos_token: bool = field(
        default = False,
        metadata = {
            "help": "whether to add a bos token at the start of each sample"
        },
    )
    use_eos_token: bool = field(
        default = True,
        metadata = {
            "help": "whether to add an eos token at the end of each sample"
        },
    )
    mask_history: bool = field(
        default = False,
        metadata = {
            "help": "whether to mask history assistant content in multi turn, if True, will only compute loss on last assistant content"
        },
    )
    online_cache_path: Optional[str] = field(
        default = None,
        metadata = {
            "help": "path to store cached dataset when training"
        },
    )
    offline_cache_path: Optional[str] = field(
        default = None,
        metadata = {
            "help": "path to cached processed dataset "
        },
    )
    tokenized_path: Optional[str] = field(
        default = None,
        metadata = {
            "help": "deprecated alias of dataset_cache_path"
        },
    )
    streaming: bool = field(
        default = False,
        metadata = {
            "help": "whether to use iterable dataset"
        },
    )
    overwrite_cache: bool = field(
        default = False,
        metadata = {
            "help": "whether to overwrite the cache of datasets"
        },
    )
    max_samples: Optional[int] = field(
        default = None,
        metadata = {
            "help": "maximum number of samples to use when training"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default = None,
        metadata = {
            "help": "number of workers to use for preprocessing"
        },
    )
    preprocessing_batch_size: Optional[int] = field(
        default = 8,
        metadata = {
            "help": "batch size to use for preprocessing"
        },
    )
    buffer_size: int = field(
        default=16384,
        metadata={"help": "Size of the buffer to randomly sample examples from in dataset streaming."},
    )
    probabilities: Optional[Union[str, list]] = field(
        default = None,
        metadata = {
            "help": "If specified, the streaming dataset is constructed by sampling examples from one source at a time according to these probabilities."
        },
    )
    stopping_strategy: str = field(
        default = "first_exhausted",
        metadata = {
            "help": "strategy to stop streaming, can be 'all_exhausted' or 'first_exhausted'"
        },
    )
    is_shared_file_system: bool = field(
        default=False,
        metadata={"help": "Whether or not to use a shared file system for the datasets."},
    )
