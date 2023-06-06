'''
Created on Aug 18, 2021

@author: xiaosonh
'''
import os
import sys
import argparse
import shutil
import math
from collections import OrderedDict

import json
import cv2
import PIL.Image
  
from sklearn.model_selection import train_test_split
from labelme import utils
import yaml


class Labelme2YOLO(object):
    
    def __init__(self, json_dir):
        self._json_dir = json_dir
        
        self._label_id_map = self._get_label_id_map(self._json_dir)
        
    def _make_train_val_dir(self):
        self._dataset_dir_path = os.path.join(self._json_dir, 'YOLODataset/')
        
        for yolo_path in (os.path.join(self._dataset_dir_path, 'train/labels'),
                          os.path.join(self._dataset_dir_path, 'val/labels')):
            if os.path.exists(yolo_path):
                shutil.rmtree(yolo_path)
            
            os.makedirs(yolo_path)
                
    def _get_label_id_map(self, json_dir):
        label_set = set()
    
        for file_name in os.listdir(json_dir):
            if file_name.endswith('json'):
                json_path = os.path.join(json_dir, file_name)
                data = json.load(open(json_path))
                for shape in data['shapes']:
                    label_set.add(shape['label'])

        # sort label_set
        label_list = list(label_set)
        label_list.sort()

        return OrderedDict([(label, label_id)
                            for label_id, label in enumerate(label_list)])

    def _train_test_split(self, folders, json_names, val_size):
        if len(folders) > 0 and 'train' in folders and 'val' in folders:
            train_folder = os.path.join(self._json_dir, 'train/')
            train_json_names = [train_sample_name + '.json'
                                for train_sample_name in os.listdir(train_folder)
                                if os.path.isdir(os.path.join(train_folder, train_sample_name))]
            
            val_folder = os.path.join(self._json_dir, 'val/')
            val_json_names = [val_sample_name + '.json'
                              for val_sample_name in os.listdir(val_folder)
                              if os.path.isdir(os.path.join(val_folder, val_sample_name))]
            
            return train_json_names, val_json_names
        
        train_idxs, val_idxs = train_test_split(range(len(json_names)), 
                                                test_size=val_size)
        train_json_names = [json_names[train_idx] for train_idx in train_idxs]
        val_json_names = [json_names[val_idx] for val_idx in val_idxs]
        
        return train_json_names, val_json_names
    
    def convert(self, val_size):
        json_names = [file_name for file_name in os.listdir(self._json_dir)
                      if os.path.isfile(os.path.join(self._json_dir, file_name)) and
                      file_name.endswith('.json')]

        train_json_names, val_json_names = train_test_split(json_names, test_size=val_size)

        self._make_train_val_dir()

        for target_dir, json_names in zip(('train/', 'val/'), (train_json_names, val_json_names)):
            for json_name in json_names:
                json_path = os.path.join(self._json_dir, json_name)
                json_data = json.load(open(json_path))

                print('Converting %s for %s ...' % (json_name, target_dir.replace('/', '')))

                yolo_obj_list = self._get_yolo_object_list(json_data)
                self._save_yolo_label(json_name, self._dataset_dir_path, target_dir + 'labels/', yolo_obj_list)
        
        print('Generating dataset.yaml file ...')
        self._save_dataset_yaml()
                
    def convert_one(self, json_name):
        json_path = os.path.join(self._json_dir, json_name)
        json_data = json.load(open(json_path))
        
        print('Converting %s ...' % json_name)
        
        yolo_obj_list = self._get_yolo_object_list(json_data)
        self._save_yolo_label(json_name, self._json_dir, '', yolo_obj_list)
    
    def _get_yolo_object_list(self, json_data):
        yolo_obj_list = []
        
        img_h, img_w = json_data['imageHeight'], json_data['imageWidth']
        for shape in json_data['shapes']:
            if shape['shape_type'] == 'circle':
                yolo_obj = self._get_circle_shape_yolo_object(shape, img_h, img_w)
            else:
                yolo_obj = self._get_other_shape_yolo_object(shape, img_h, img_w)
            
            yolo_obj_list.append(yolo_obj)
            
        return yolo_obj_list
    
    def _get_circle_shape_yolo_object(self, shape, img_h, img_w):
        obj_center_x, obj_center_y = shape['points'][0]
        
        radius = math.sqrt((obj_center_x - shape['points'][1][0]) ** 2 + 
                           (obj_center_y - shape['points'][1][1]) ** 2)
        obj_w = 2 * radius
        obj_h = 2 * radius
        
        yolo_center_x= round(float(obj_center_x / img_w), 6)
        yolo_center_y = round(float(obj_center_y / img_h), 6)
        yolo_w = round(float(obj_w / img_w), 6)
        yolo_h = round(float(obj_h / img_h), 6)
            
        label_id = self._label_id_map[shape['label']]
        
        return label_id, yolo_center_x, yolo_center_y, yolo_w, yolo_h
    
    def _get_other_shape_yolo_object(self, shape, img_h, img_w):
        def __get_object_desc(obj_port_list):
            def __get_dist(int_list): return max(int_list) - min(int_list)

            x_lists = [port[0] for port in obj_port_list]
            y_lists = [port[1] for port in obj_port_list]
            
            return min(x_lists), __get_dist(x_lists), min(y_lists), __get_dist(y_lists)
        
        obj_x_min, obj_w, obj_y_min, obj_h = __get_object_desc(shape['points'])
                    
        yolo_center_x= round(float((obj_x_min + obj_w / 2.0) / img_w), 6)
        yolo_center_y = round(float((obj_y_min + obj_h / 2.0) / img_h), 6)
        yolo_w = round(float(obj_w / img_w), 6)
        yolo_h = round(float(obj_h / img_h), 6)
            
        label_id = self._label_id_map[shape['label']]
        
        return label_id, yolo_center_x, yolo_center_y, yolo_w, yolo_h
    
    def _save_yolo_label(self, json_name, label_dir_path, target_dir, yolo_obj_list):
        txt_path = os.path.join(label_dir_path, 
                                target_dir, 
                                json_name.replace('.json', '.txt'))

        with open(txt_path, 'w+') as f:
            for yolo_obj_idx, yolo_obj in enumerate(yolo_obj_list):
                yolo_obj_line = '%s %s %s %s %s\n' % yolo_obj \
                    if yolo_obj_idx + 1 != len(yolo_obj_list) else \
                    '%s %s %s %s %s' % yolo_obj
                f.write(yolo_obj_line)
                
    def _save_yolo_image(self, json_data, json_name, image_dir_path, target_dir):
        img_name = json_name.replace('.json', '.png')
        img_path = os.path.join(image_dir_path, target_dir,img_name)
        
        if not os.path.exists(img_path):
            img = utils.img_b64_to_arr(json_data['imageData'])
            PIL.Image.fromarray(img).save(img_path)
        
        return img_path
    
    def _save_dataset_yaml(self):
        dataset_yaml = {
            'train': os.path.join(self._dataset_dir_path, 'train/images'),
            'val': os.path.join(self._dataset_dir_path, 'val/images'),
            'nc': len(self._label_id_map),
            'names': list(self._label_id_map.keys())
        }

        with open(os.path.join(self._dataset_dir_path, 'dataset.yaml'), 'w') as yaml_file:
            yaml.dump(dataset_yaml, yaml_file, default_flow_style=False)

        print('Dataset.yaml file saved in', self._dataset_dir_path)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_dir',type=str,
                        help='Please input the path of the labelme json files.')
    parser.add_argument('--val_size',type=float, nargs='?', default=None,
                        help='Please input the validation dataset size, for example 0.1 ')
    parser.add_argument('--json_name',type=str, nargs='?', default=None,
                        help='If you put json name, it would convert only one json file to YOLO.')
    args = parser.parse_args(sys.argv[1:])

    # for debug
    json_dir = "/home/kvnptl/work/b_it_bots/b_it_bot_work/2d_object_detection/robocup_2023_dataset/dataset_collection/vamsi_shubham/set2_155_307_annotated"
    convertor = Labelme2YOLO(json_dir)
    convertor.convert(val_size=0.1)

    # convertor = Labelme2YOLO(args.json_dir)

    # if args.json_name is None:
    #     convertor.convert(val_size=args.val_size)
    # else:
    #     convertor.convert_one(args.json_name)
