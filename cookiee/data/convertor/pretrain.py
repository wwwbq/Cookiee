from .utils import convert_images, convert_domain
from ..spec import DatasetSpec

class PretrainConvertor:
    def convert_pretrain(self, example, dataset_spec: DatasetSpec):
        r"""
        Converts pretrain format dataset to the standard format.
        """

        assert dataset_spec.text in example

        system = example[dataset_spec.system] if dataset_spec.system else None
        prompt = example[dataset_spec.text]
        response = ""

        return {
            "_system": system,
            "_prompt": prompt,
            "_response": response,
        }


    def __call__(self, example, dataset_spec: DatasetSpec):
        outputs = self.convert_pretrain(example, dataset_spec)
        images = convert_images(example, dataset_spec)
        domain = convert_domain(example, dataset_spec)

        outputs.update({"_images": images, "_domain": domain})

        return outputs
    