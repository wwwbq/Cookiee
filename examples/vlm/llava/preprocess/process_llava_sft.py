import os
import pandas as pd
from tqdm import tqdm
from helper import read, save
from datasets import load_dataset
from datasets.arrow_dataset import Dataset as HFDataset

path = "/root/paddlejob/share-storage/gpfs/wangbingquan/hf_hub/llava/LLaVA-Instruct-150K/llava_v1_5_mix665k.json"
save_path = "/root/paddlejob/share-storage/gpfs/wangbingquan/hf_hub/llava/LLaVA-Instruct-150K/llava_v1_5_mix665k_clean.json"

datasets = read(path)

saved_datasets = []
for i in tqdm(range(len(datasets))):
    datasets[i].pop('id')
    saved_datasets.append(datasets[i])

save(saved_datasets, save_path)

# df = pd.read_json(save_path)
# dataset = HFDataset.from_pandas(df, preserve_index=False)

# num_cleaned_files = dataset.cleanup_cache_files()
# print(f"成功清理了 {num_cleaned_files} 个缓存文件。")


def convert_images_to_jpg(path):
    """
    将指定路径下所有支持的非 JPG 格式图片转换为 JPG 格式。

    支持的格式包括: .png, .gif, .bmp, .tiff, .webp 等。
    """
    # 1. 定义支持转换的非 JPG 图片格式列表（可以根据需要增删）
    from PIL import Image

    supported_formats = ('.png', '.gif', '.bmp', '.tiff', '.tif', '.webp', '.jpeg')
    
    converted_count = 0
    
    print(f"开始扫描文件夹: {path}")
    
    # 使用 tqdm 创建进度条
    for filename in tqdm(os.listdir(path)):
        # 2. 将文件名转为小写，并检查其后缀是否在支持的格式列表中
        if filename.lower().endswith(supported_formats):
            
            original_path = os.path.join(path, filename)
            
            # 3. 构建新的 JPG 文件名和路径
            # 使用 os.path.splitext 分离主干和后缀，确保正确替换
            file_basename = os.path.splitext(filename)[0]
            jpg_filename = file_basename + '.jpg'
            jpg_path = os.path.join(path, jpg_filename)
            
            # 防止文件被重复转换（例如 a.jpeg -> a.jpg）
            if original_path == jpg_path:
                continue

            try:
                # 4. 使用 with 语句安全地打开和处理图片
                with Image.open(original_path) as img:
                    # 5. 核心转换逻辑：将图片转换为 'RGB' 模式
                    # PNG 等格式可能有 'RGBA'（带透明通道），转换为 'RGB' 会移除透明度
                    # 这是保存为 JPG 的必要步骤，因为 JPG 不支持透明通道
                    with img.convert('RGB') as rgb_img:
                        # 6. 保存为 JPG 格式
                        rgb_img.save(jpg_path, 'jpeg')
                
                converted_count += 1

            except Exception as e:
                # 增加异常处理，防止因个别损坏文件导致程序中断
                print(f"\n处理文件 {filename} 时出错: {e}")

    print(f"\n处理完成！总共成功转换了 {converted_count} 张图片。")
    return converted_count

ocr_vqa_path = "/root/paddlejob/share-storage/gpfs/wangbingquan/hf_hub/llava/llava_sft_images/ocr_vqa/images"
convert_images_to_jpg(ocr_vqa_path)