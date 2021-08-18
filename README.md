# Labelme2YOLO

Convert LabelMe Annotation Tool JSON format to YOLO text file format.

How to use:
Input LabelMe JSON folder as parameter for --json_dir, and define validation dataset size as parameter for --val_size.

Run python by below command.
```bash
python labelme2yolo.py --json_dir /home/username/labelme_json_dir/ --val_size 0.2
```
Script would generate YOLO format dataset labels and images under different folders, for example:
```bash
/home/username/labelme_json_dir/YOLODataset/labels/train/
/home/username/labelme_json_dir/YOLODataset/labels/val/
/home/username/labelme_json_dir/YOLODataset/images/train/
/home/username/labelme_json_dir/YOLODataset/images/val/
```
