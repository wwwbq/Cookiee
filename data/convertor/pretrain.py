from .utils import DatasetAttr, convert_images, convert_channel

class PretrainConvertor:
    def convert_pretrain(self, example, dataset_attr: DatasetAttr):
        r"""
        Converts pretrain format dataset to the standard format.
        """

        assert dataset_attr.text in example

        system = example[dataset_attr.system] if dataset_attr.system else None
        prompt = example[dataset_attr.text]
        response = ""

        return {
            "_system": system,
            "_prompt": prompt,
            "_response": response,
        }


    def __call__(self, example, dataset_attr: DatasetAttr):
        outputs = self.convert_pretrain(example, dataset_attr)
        images = convert_images(example, dataset_attr)
        channel = convert_channel(example, dataset_attr)

        outputs.update({"_images": images, "_channel": channel})

        return outputs
    