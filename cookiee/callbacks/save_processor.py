import os
from typing_extensions import override
from transformers import ProcessorMixin, TrainerCallback, BaseImageProcessor
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers import TrainerControl, TrainerState, TrainingArguments

class SaveProcessorCallback(TrainerCallback):
    r"""A callback for saving the processor."""

    def __init__(self, processor: "ProcessorMixin", image_processor: "BaseImageProcessor") -> None:
        self.processor = processor
        self.image_processor = image_processor

    @override
    def on_save(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if args.should_save:
            output_dir = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
            self.processor.save_pretrained(output_dir)
            self.image_processor.save_pretrained(output_dir)

    @override
    def on_train_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if args.should_save:
            self.processor.save_pretrained(args.output_dir)
            self.image_processor.save_pretrained(args.output_dir)