# cvat2yolo

Convert the dataset downloaded from the CVAT platform into the format required for YOLO training

## 

- Parse `annotation.xml` downloaded from CVAT
- Automatically generate label files of YOLO format, and each image correspond to a txt 
- Support multi-classes mapping
- Create a configuration file named `data.yml`, supporting for model training directly in YOLOv8 and higher versions

## Install
```bash
git clone https://github.com/DDAshley126/cvat2yolo.git
cd cvat2yolo
pip install requirements.txt
```

## Description

The repository composes of following key components:

- converter.py: a script for processing XML comments and converting them into the format required by YOLO
- class_config.json: a json file, including configuration file of detecting classes
- config.yml: example configuration of YOLO model training
- annotation.xml: example of annotating images in CVAT