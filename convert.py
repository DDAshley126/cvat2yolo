import shutil
from pathlib import Path
import random
import argparse
from typing import Tuple


def split_yolo_dataset(src_dir: str, ratios: Tuple = (0.6, 0.2, 0.2)):
    """
    划分训练集验证集测试集

    Args：
        src_dir: 原始数据集目录（需要包含images和labels子目录）
        ratios: 训练/验证/测试集比例
    """
    dst_path = Path(f'{src_dir}_split')
    if dst_path.exists():
        shutil.rmtree(dst_path)

    for i, r in enumerate(ratios):
        if not isinstance(r, (int, float)):
            raise TypeError(f'Parameter requires float or int, but got {type(r)}')
    if sum(ratios) != 1:
        raise ValueError(f'Sum of ratios must be equal to 1')
    if len(ratios) != 3:
        raise ValueError(f'Ratios must contain 3 elements')

    (dst_path / 'images').mkdir(parents=True, exist_ok=True)
    (dst_path / 'labels').mkdir(parents=True, exist_ok=True)

    image_list = (Path(src_dir) / 'images').iterdir()
    image_list = [image for image in image_list if image.suffix in ['.png', '.jpg', 'jpeg']]
    num = len(image_list)
    train_num = int(ratios[0] * num)
    val_num = int(ratios[1] * num)

    random.shuffle(image_list)
    dataset_split = {
        'train': image_list[:train_num],
        'val': image_list[train_num:train_num + val_num],
        'test': image_list[train_num + val_num:]
    }

    for split in dataset_split:
        (dst_path / "images" / split).mkdir()
        (dst_path / "labels" / split).mkdir()

    for split, images in dataset_split.items():
        for image in images:
            shutil.copy(image, Path(dst_path) / 'images' / split)

            label_txt = image.name.split('.')[0]
            shutil.copyfile(Path(src_dir) / 'labels' / f'{label_txt}.txt',
                            Path(dst_path) / 'labels' / split / f'{label_txt}.txt')

    print(f'转换完成，数据集已保存至{dst_path}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    subparser = parser.add_subparsers(title='subfunc')

    parser_split = subparser.add_parser('split_yolo_dataset', help='将CVAT标注转换为YOLO格式')
    parser_split.add_argument("--src_dir", type=str, required=True, help="CVAT标注数据集目录，包含images和annotations子目录")
    parser_split.add_argument('--ratios', type=float, nargs=3, default=(0.6, 0.2, 0.2))
    parser_split.set_defaults(func=split_yolo_dataset)

    args = parser.parse_args()
    
    if hasattr(args, 'func'):
        func_args = {k: v for k, v in vars(args).items() if k != 'func'}
        args.func(**func_args)
    else:
        parser.print_help()