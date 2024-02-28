# Labelme2YOLO

Help converting LabelMe Annotation Tool JSON format to YOLO text file format. 
If you've already marked your segmentation dataset by LabelMe, it's easy to use this tool to help converting to YOLO format dataset.

## Parameters Explain
**--json_dir** LabelMe JSON files folder path.

**--val_size (Optional)** Validation dataset size, for example 0.2 means 20% for validation and 80% for training. Default value is 0.1 .

**--json_name (Optional)** Convert single LabelMe JSON file.

**--seg (Optional)** Convert to [YOLOv5 v7.0](https://github.com/ultralytics/yolov5/tree/v7.0#segmentation--new) instance segmentation dataset.

## how to install the environment
1. clone the repo and go to the folder
```
git clone https://github.com/linwang9926/Labelme2YOLO.git
cd Labelme2YOLO
```

2. create a new conda environment
```
conda create -n labelme2yolo python=3.9
```

3. activate the environment and install the requirements
```
conda activate labelme2yolo
pip install -r requirements.txt -q
```

## How to Use

### 1. Convert JSON files, split training and validation dataset by --val_size
Put all LabelMe JSON files under **labelme_json_dir**, and run this python command.
```bash
python labelme2yolo.py --json_dir /home/username/labelme_json_dir/ --val_size 0.2
```
Script would generate YOLO format dataset labels and images under different folders, for example,
```bash
# when specifying `--seg', "YOLODataset" will be "YOLODataset_seg"
/home/username/labelme_json_dir/YOLODataset/labels/train/
/home/username/labelme_json_dir/YOLODataset/labels/val/
/home/username/labelme_json_dir/YOLODataset/images/train/
/home/username/labelme_json_dir/YOLODataset/images/val/

/home/username/labelme_json_dir/YOLODataset/dataset.yaml
```

### 2. Convert JSON files, split training and validation dataset by folder
If you already split train dataset and validation dataset for LabelMe by yourself, please put these folder under labelme_json_dir, for example,
```bash
/home/username/labelme_json_dir/train/
/home/username/labelme_json_dir/val/
```
Put all LabelMe JSON files under **labelme_json_dir**. 
Script would read train and validation dataset by folder.
Run this python command.
```bash
python labelme2yolo.py --json_dir /home/username/labelme_json_dir/
```
Script would generate YOLO format dataset labels and images under different folders, for example,
```bash
# when specifying `--seg', "YOLODataset" will be "YOLODataset_seg"
/home/username/labelme_json_dir/YOLODataset/labels/train/
/home/username/labelme_json_dir/YOLODataset/labels/val/
/home/username/labelme_json_dir/YOLODataset/images/train/
/home/username/labelme_json_dir/YOLODataset/images/val/

/home/username/labelme_json_dir/YOLODataset/dataset.yaml
```

### 3. Convert single JSON file
Put LabelMe JSON file under **labelme_json_dir**. , and run this python command.
```bash
python labelme2yolo.py --json_dir /home/username/labelme_json_dir/ --json_name 2.json
```
Script would generate YOLO format text label and image under **labelme_json_dir**, for example,
```bash
/home/username/labelme_json_dir/2.text
/home/username/labelme_json_dir/2.png
```

##
Only tested on Centos 7/Python 3.6 environment.
