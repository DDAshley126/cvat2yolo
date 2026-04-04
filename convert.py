import os
import shutil
from pathlib import Path
import random
import argparse


def split_yolo_dataset(src_dir, ratios=(0.6, 0.2, 0.2)):
    """
    划分训练集验证集测试集

    Args：
        src_dir: 原始数据集目录(需要包含images和labels)
        ratios: 训练集:验证集:测试集比例，默认6：2：2
    """
    dst_dir = f"{src_dir}_split"
    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)

    # 创建标准目录结构
    base_path = Path(dst_dir)
    (base_path / "images").mkdir(parents=True)
    (base_path / "labels").mkdir(parents=True)

    # 复制原始文件
    shutil.copytree(Path(src_dir) / "images", base_path / "images" / "original")
    shutil.copytree(Path(src_dir) / "labels", base_path / "labels" / "original")
    shutil.copy(Path(src_dir) / "classes.txt", base_path)
    if (Path(src_dir) / "notes.json").exists():
        shutil.copy(Path(src_dir) / "notes.json", base_path)

    # 获取所有图像文件名（不带扩展名）
    all_images = [f.stem for f in (base_path / "images/original").glob("*.*")
                  if f.suffix.lower() in ['.jpg', '.png', '.jpeg']]
    random.shuffle(all_images)  # 随机打乱顺序

    # 计算分割点
    total = len(all_images)
    train_end = int(ratios[0] * total)
    val_end = train_end + int(ratios[1] * total)

    # 划分数据集
    splits = {
        "train": all_images[:train_end],
        "val": all_images[train_end:val_end],
        "test": all_images[val_end:]
    }

    # 创建目标目录结构
    for split in splits:
        (base_path / "images" / split).mkdir()
        (base_path / "labels" / split).mkdir()

    # 移动文件到对应目录
    for split, files in splits.items():
        for fname in files:
            # 处理图像文件
            src_img = next((base_path / "images/original").glob(f"{fname}.*"))
            dst_img = base_path / "images" / split / src_img.name
            shutil.move(str(src_img), str(dst_img))

            # 处理标注文件
            src_label = base_path / "labels/original" / f"{fname}.txt"
            dst_label = base_path / "labels" / split / src_label.name
            if src_label.exists():
                shutil.move(str(src_label), str(dst_label))
            else:
                print(f"警告：缺失标注文件 {src_label}")

    # 清理原始目录
    shutil.rmtree(base_path / "images/original")
    shutil.rmtree(base_path / "labels/original")

    print(f"数据集已分割到 {dst_dir}")
    print(f"最终目录结构：")
    print(f"images/")
    print(f"├── train/ : {len(splits['train'])} 图像")
    print(f"├── val/   : {len(splits['val'])} 图像")
    print(f"└── test/  : {len(splits['test'])} 图像")
    print(f"labels/")
    print(f"├── train/ : {len(splits['train'])} 标注")
    print(f"├── val/   : {len(splits['val'])} 标注")
    print(f"└── test/  : {len(splits['test'])} 标注")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将CVAT标注转换为YOLO格式")
    parser.add_argument("--cvat_dir", type=str, required=True, help="CVAT标注数据集目录，包含images和annotations子目录")
    args = parser.parse_args()
