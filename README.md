# Labelme2YOLO

Convert LabelMe Annotation Tool JSON format to YOLO text file format.

How to use:
Input LabelMe JSON folder as parameter for --json_dir, and define validation dataset size as parameter for --val_size.

Run python by below command.
```bash
python labelme2yolo.py --json_dir /home/username/labelme_json_dir/ --val_size 0.2
```
