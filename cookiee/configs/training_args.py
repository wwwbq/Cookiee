from typing import Optional
from dataclasses import dataclass, field

from transformers import TrainingArguments as HFTrainingArguments


@dataclass
class TrainingArguments(HFTrainingArguments):
    remove_unused_columns: Optional[bool] = field(
        default=False, 
        metadata={"help": "Overwrite defualt value to False. Remove columns not required by the model when using an nlp.Dataset."}
    )
    
