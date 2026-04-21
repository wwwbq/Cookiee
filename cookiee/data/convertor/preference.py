from .utils import convert_images, convert_domain
from ..spec import DatasetSpec

class PreferenceConvertor:
    def convert_preference(self, example, dataset_spec: DatasetSpec):
        if dataset_spec.format == "content":
            assert dataset_spec.chosen in example and dataset_spec.rejected in example
            system = example[dataset_spec.system] if dataset_spec.system else None
            chosen = example[dataset_spec.chosen]
            rejected = example[dataset_spec.rejected]
            return {
                "_system": system,
                "_prompt": "", # 兼容未来可能的 偏好-指令 数据的格式
                "_chosen": chosen,
                "_rejected": rejected
            }
        else:
            raise NotImplementedError("Preference dataset only support pretrain format.")


    def __call__(self, example, dataset_spec: DatasetSpec):
        outputs = self.convert_preference(example, dataset_spec)
        images = convert_images(example, dataset_spec)
        domain = convert_domain(example, dataset_spec)

        outputs.update({"_images": images, "_domain": domain})

        return outputs
    