from .pretrain import PretrainConvertor
from .sft import SftConvertor
from .utils import convert_images, convert_domain
from ..spec import DatasetSpec

class MidtrainConvertor:
    tokenizer = None
    pretrain_convertor = PretrainConvertor()
    sft_convertor = SftConvertor()


    def convert_midtrain(self, example, dataset_spec: DatasetSpec):
        r"""
        Converts midtrain format dataset to the standard format.
        """
        if dataset_spec.format == "content":
            return self.pretrain_convertor.convert_pretrain(example, dataset_spec)
        
        if dataset_spec.format == "alpaca":
            outputs = self.sft_convertor.convert_alpaca(example, dataset_spec)
        elif dataset_spec.format == "sharegpt":
            outputs = self.sft_convertor.convert_sharegpt(example, dataset_spec)
        else:
            raise ValueError("Unsupported formatting in Midtrain: {}".format(dataset_spec.format))
        
        prompts = outputs["_prompt"]
        responses = outputs["_response"]

        if dataset_spec.midtrain_template == "sft":
            assert self.tokenizer is not None
            prompts = self.tokenizer.apply_chat_template(prompts, add_generation_prompt=True, tokenize=False)
            outputs["_prompt"] = prompts + responses[0]["content"]
            outputs["_response"] = ""

        elif dataset_spec.midtrain_template == "plain":
            messages = prompts + responses
            outputs["_prompt"] = "\n".join([f"{message['role']}: {message['content']}" for message in messages])
            outputs["_response"] = ""

        elif dataset_spec.midtrain_template == "none":
            messages = prompts + responses
            outputs["_prompt"] = "\n".join([f"{message['content']}" for message in messages])
            outputs["_response"] = ""

        return outputs


    def __call__(self, example, dataset_spec: DatasetSpec):
        outputs = self.convert_midtrain(example, dataset_spec)
        images = convert_images(example, dataset_spec)
        domain = convert_domain(example, dataset_spec)

        outputs.update({"_images": images, "_domain": domain})

        return outputs
    