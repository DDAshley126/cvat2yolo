# cvat2yolo 

将CVAT平台下载的数据集转换为YOLO训练所需的格式

## 功能特点

- 解析CVAT导出的`annotation.xml`文件
- 自动生成yolo格式的标签文件（每张图片对应一个txt）
- 支持多类别映射
- 生成`data.yml`配置文件，可直接用于YOLOv8以及上版本的模型训练

## 安装方式
```bash
git clone https://gitee.com/DDAshley126/cvat2yolo.git
cd cvat2yolo
pip install requirements.txt
```

## 描述

该项目包括关键组成部分：

- converter.py：一个用于处理XML注释并将其转换为YOLO所需格式的Python脚本。
- class_config.json：一个JSON文件，其中包含检测类的配置信息
- config.yml：用于训练YOLO模型的示例配置文件
- annotation.xml：cvat格式标注图像的示例