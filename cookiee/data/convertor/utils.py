import os
from ..spec import DatasetSpec
    

def convert_images(example, dataset_spec: DatasetSpec):
    if dataset_spec.images is None:
        return None

    images = example[dataset_spec.images]

    if images is None or len(images) == 0:
        return None
    
    images = images[:]
    if not isinstance(images, list):
        images = [images]
    
    for i in range(len(images)):
        if isinstance(images[i], str):
            if dataset_spec.image_folder:
                image_folder = dataset_spec.image_folder
            else:
                if dataset_spec.folder:
                    image_folder = dataset_spec.folder
                else:
                    image_folder = os.path.dirname(dataset_spec.file) if os.path.isfile(dataset_spec.file) else dataset_spec.file
            images[i] = os.path.join(image_folder, images[i])

    return images


def convert_domain(example, dataset_spec: DatasetSpec):
    # 优先级：整个数据集的统一domain > 单条数据的domain
    if dataset_spec.domain is not None:
        return dataset_spec.domain
    else:
        return example.get("domain", None)