import shutil
from pathlib import Path
import random
import os
from typing import Tuple, Literal
import yaml
import xml.etree.ElementTree as ET
from tqdm import tqdm
import cv2
from xml.dom import minidom
from ultralytics import YOLO
import json


def split_yolo_dataset(src_dir: str, ratios: Tuple = (0.6, 0.2, 0.2)):
    """
    Split dataset

    Args：
        src_dir: Initial dataset directory, images and labels must bo included
        ratios: Train/Validation/Test ratio
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


def cvat_detect_to_yolo_detect(cvat_dir, output_dir, cvat_save_images=True, image_dir=None):
    """
    Convert the annotation files downloaded from CVAT in Ultralytics YOLO Detection 1.0 format to the standard YOLO format.

    Args:
        cvat_dir: Directory path of Ultralytics YOLO Detection 1.0 downloaded from CVAT
        output_dir: Output directory
        cvat_save_images: Choose 'save imagess' when downloading dataset from CVAT, otherwise set False
        image_dir: Images directory
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)

    # Classes names
    with open(os.path.join(cvat_dir, 'data.yaml'), 'r', encoding='utf-8') as f:
        result = yaml.load(f.read(), Loader=yaml.FullLoader)
    class_names = [result['names'][i] for i in result['names']]

    # Adjusting the structure of the images and labels folders
    if cvat_save_images:
        old_images_path = os.path.join(cvat_dir, 'images', 'train')
        new_images_path = os.path.join(cvat_dir, 'images')
        for i in os.listdir(old_images_path):
            os.rename(os.path.join(old_images_path, i), os.path.join(new_images_path, i))
    else:
        if image_dir is None:
            print(f'Lacking of parameter: {image_dir}')
        else:
            # Copy the images according to the annotation results, avoiding the inclusion of unannotated images
            image_list = os.listdir(image_dir)
            exist_image_list = []
            for exist_image in os.listdir(os.path.join(cvat_dir, 'labels', 'train')):
                for image in image_list:
                    if image.startswith(exist_image.split('.')[0]):
                        exist_image_list.append(image)
            for image in exist_image_list:
                shutil.copy(os.path.join(image_dir, image), os.path.join(output_dir, 'images', image))

            # Move labels
            shutil.copytree(os.path.join(cvat_dir, 'labels', 'train'), os.path.join(output_dir, 'labels'))
    print('Transform successfully!')


def cvat_xml_to_yolo_pose(cvat_dir, class_names: dict, output_dir, cvat_save_images=True, image_dir=None, key_point_fromat: Literal['2D', '3D'] = '2D'):
    """
    Convert CVAT annotations.xml file to YOLO pose format

    Args:
        cvat_dir: Directory of CVAT exported annotations.xml
        class_names: Classes names, in the same order as those on CVAT (currently, the XML exported by CVAT does not record category files).
        output_dir: Output directory
        cvat_save_images: Choose 'save imagess' when downloading dataset from CVAT, otherwise set False
        image_dir: Images directory
    """
    output_labels_dir = os.path.join(output_dir, 'labels')
    output_images_dir = os.path.join(output_dir, 'images')
    os.makedirs(output_labels_dir, exist_ok=True)
    os.makedirs(output_images_dir, exist_ok=True)

    # Parse XML
    xml_path = os.path.join(cvat_dir, 'annotations.xml')
    doc = ET.parse(xml_path)
    root = doc.getroot()
    version = root.find('version')

    for image in root.findall('image'):
        name = image.get('name')
        image_width = int(image.get('width'))
        image_height = int(image.get('height'))

        # Get box border
        boxs = image.findall('box')
        for box in boxs:
            xtl = float(box.get('xtl'))
            ytl = float(box.get('ytl'))
            xbr = float(box.get('xbr'))
            ybr = float(box.get('ybr'))

            # Coordinate normalization calculation
            box_width = xbr - xtl
            box_height = ybr - ytl
            norm_x_center = (xtl + box_width / 2) / image_width
            norm_y_center = (ytl + box_height / 2) / image_height
            norm_width = box_width / image_width
            norm_height = box_height / image_height

        # Get keypoints
        keypoints = []
        points = image.findall('points')
        for point in points:
            point_label = point.get('label')
            point_location = point.get('points')
            [x, y] = point_location.split(',')
            norm_x_point = float(x) / image_width
            norm_y_point = float(y) / image_width

            # Find the class belongs to
            point_index = [key for key, val in class_names['kpt_names'].items() if val == point_label][0]
            point_info = [point_index, norm_x_center, norm_y_center, norm_width, norm_height, norm_x_point, norm_y_point]
            keypoints.extend([point_info])

        # Write txt
        if keypoints:
            name = name.split('.')[0]
            txt_path = os.path.join(output_dir, 'labels', f'{name}.txt')
            with open(txt_path, 'w') as f:
                # Format of 2D: <class-index> <x> <y> <width> <height> <px1> <py1> <px2> <py2> ... <pxn> <pyn>
                for i in keypoints:
                    f.write(f'{i[0]} {i[1]} {i[2]} {i[3]} {i[4]} {i[5]} {i[6]} 0 0\n')

    # Copy the images according to the annotation results, avoiding the inclusion of unannotated images
    if cvat_save_images:
        old_images_path = os.path.join(cvat_dir, 'images', 'train')
        new_images_path = os.path.join(cvat_dir, 'images')
        for i in os.listdir(old_images_path):
            os.rename(os.path.join(old_images_path, i), os.path.join(new_images_path, i))
    else:
        if image_dir is None:
            print(f'Lacking of parameter: {image_dir}')
        else:
            image_list = os.listdir(os.path.join(output_dir, 'labels'))
            exist_image_list = []
            for exist_image in image_list:
                for image in os.listdir(image_dir):
                    if image.startswith(exist_image.split('.')[0]):
                        exist_image_list.append(image)
            for image in exist_image_list:
                shutil.copy(os.path.join(image_dir, image), os.path.join(output_dir, 'images', image))

    print(f"Transform successfully!")


def yolo_to_cvat(image_dir, label_dir, output_dir, class_names):
    """
    Convert YOLO predict result txt files to CVAT 1.1 format. It is useful after transformation to XML format, it can be directly uploaded to CVAT.

    Args:
        image_dir: Directory of storing images locaally, like './images' 
        label_dir: Directory of storing YOLO txt labels files, like './labels'
        output_dir: Directory of outputting CVAT 1.1
        class_names: List of classes names, with the index corresponding to the classes' ID, like ['frame', 'wheel', 'seat', ...]
    """
    os.makedirs(output_dir, exist_ok=True)
    images_output_dir = os.path.join(output_dir, 'images')
    os.makedirs(images_output_dir, exist_ok=True)

    # Create root element
    root = ET.Element('annotations')
    root.set('version', '1.0')

    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    image_id = 0

    for img_file in tqdm(image_files, desc="Processing..."):
        img_path = os.path.join(image_dir, img_file)
        img_output_path = os.path.join(images_output_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: cannot read image {img_file}, pass")
            continue
        height, width, depth = img.shape
        cv2.imwrite(img_output_path, img)

        # Find the corresponding YOLO label file
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(label_dir, label_file)

        # Generate image element
        image_elem = ET.SubElement(root, 'image')
        image_elem.set('id', str(image_id))
        image_elem.set('name', f'images/{img_file}')
        image_elem.set('width', str(width))
        image_elem.set('height', str(height))

        # Add box element
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue  # Skip with incorrect format
                    class_id, x_center, y_center, w_norm, h_norm = map(float, parts)
                    class_id = int(class_id)

                    # Convert YOLO format (normalized center xywh) to CVAT format (absolute coordinates xywh)
                    abs_x = (x_center - w_norm / 2) * width
                    abs_y = (y_center - h_norm / 2) * height
                    abs_w = w_norm * width
                    abs_h = h_norm * height

                    # Generate box
                    box_elem = ET.SubElement(image_elem, 'box')
                    box_elem.set('label', class_names[class_id])
                    box_elem.set('occluded', '0')
                    box_elem.set('source', 'manual')
                    box_elem.set('xtl', str(abs_x))
                    box_elem.set('ytl', str(abs_y))
                    box_elem.set('xbr', str(abs_x + abs_w))
                    box_elem.set('ybr', str(abs_y + abs_h))
        image_id += 1

    xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent='  ')
    xml_output_path = os.path.join(output_dir, 'annotations.xml')
    with open(xml_output_path, 'w', encoding='utf-8') as f:
        f.write(xml_str)

    print(f"Transform successfully.")


def yolo_txt_to_cvat_xml(image_dir: str, txt_dir: str, output_dir: str, class_names: dict):
    """
    Predict images and save the prediction result files, then convert YOLO format files to CVAT files,
    making it easier to upload to CVAT for further annotation and simplifying the annotation work.
    There is no need to build a CVAT model prediction service (cause I always fail to set the service).

    Args:
        image_dir: Directory of images, locally
        txt_dir: Directory for storing YOLO prediction .txt files
        output_dir: Directory for exporting CVAT format files
        class_names: Class mapping dict, for example {0: 'person', 1: 'car', 2: 'road'}

    Example:
        Step 1: Save prediction results
        ```
            model = YOLO('your model path')
            result = model.predict(source='images directory', save_txt=True)
        ```
        Step 2: Run it
        ```
            yolo_txt_to_cvat_xml(image_dir, txt_dir, output_dir, class_names)
        ```

        Step 3: Upload to CVAT
        In CVAT, choose 'Import annotations' to upload annotations.xml. Remember, choose 'CVAT 1.1' in 'Import format'.
    """
    image_extensions = ('jpg', 'png', 'jpeg')
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(Path(image_dir).glob(f'*{ext}'))
        image_paths.extend(Path(image_dir).glob(f'*{ext.upper()}'))

    if not image_paths:
        raise ValueError(f'Error: {image_dir} cannot found image files.')

    # Create root element
    root = ET.Element('annotations')

    for img_path in sorted(image_paths):
        img_name = img_path.name
        stem = img_path.stem

        # Get image size
        img = cv2.imread(str(img_path))
        if img is None:
            print(f'Warning: cannot read {img_path}, pass')
            continue
        width = img.shape[1]
        height = img.shape[0]

        # Find YOLO txt file
        txt_path = Path(txt_dir) / f'{stem}.txt'
        if not txt_path.exists():
            print(f'Warning: cannot found {txt_path}, pass')

        # Create image element
        image_elem = ET.SubElement(root, 'images')
        image_elem.set('name', img_name)
        image_elem.set('width', str(width))
        image_elem.set('height', str(height))

        with open(txt_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                print(f'Warning: {txt_path} error.')
                continue
            id = int(parts[0])
            xc = float(parts[1])
            yc = float(parts[2])
            bw = float(parts[3])
            bh = float(parts[4])

            # Convert to absolute coordinates (xtl, ytl, xbr, ybr)
            xtl = (xc - bw / 2) * width
            ytl = (yc - bh / 2) * height
            xbr = (xc + bw / 2) * width
            ybr = (yc + bh / 2) * height

            label = class_names.get(id, str(id))

            # Add box element
            box_elem = ET.SubElement(image_elem, "box")
            box_elem.set("label", label)
            box_elem.set("xtl", f"{xtl:.2f}")
            box_elem.set("ytl", f"{ytl:.2f}")
            box_elem.set("xbr", f"{xbr:.2f}")
            box_elem.set("ybr", f"{ybr:.2f}")
            box_elem.set("occluded", "0")
            box_elem.set("source", "auto")

    output_path = Path(output_dir) / 'annotations.xml'
    rough = ET.tostring(root, encoding='utf-8')
    parsed = minidom.parseString(rough)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(parsed.toprettyxml(indent='  '))

    print(f"Transform successfully.")