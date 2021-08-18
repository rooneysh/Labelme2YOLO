'''
Created on Aug 18, 2021

@author: rooneysh
'''
import os
import sys
import argparse

import json
import cv2
import PIL.Image
 
from sklearn.model_selection import train_test_split
from labelme import utils

class Labelme2YOLO(object):
    
    def __init__(self, json_dir):
        self._json_dir = json_dir
        
        self._label_id_map = self._get_label_id_map(self._json_dir)
        
        self._label_dir_path = os.path.join(json_dir, 'YOLODataset/labels/')
        self._image_dir_path = os.path.join(json_dir, 'YOLODataset/images/')
        
        self._make_train_val_dir(self._json_dir)
        
    def _make_train_val_dir(self):
        for yolo_path in (os.path.join(self._label_dir_path + 'train/'), 
                          os.path.join(self._label_dir_path + 'val/'),
                          os.path.join(self._image_dir_path + 'train/'), 
                          os.path.join(self._image_dir_path + 'val/')):
            if not os.path.exists(yolo_path):
                os.makedirs(yolo_path)
                
    def _get_label_id_map(self, json_dir):
        label_set = set()
    
        for file_name in os.listdir(json_dir):
            if file_name.endswith('json'):
                json_path = os.path.join(json_dir, file_name)
                data = json.load(open(json_path))
                for shape in data['shapes']:
                    label_set.add(shape['label'])
        
        return {label: label_id for label_id, label in enumerate(label_set)}
    
    def _train_test_split(self, folders, json_names, val_size):
        if len(folders) > 0 and 'train' in folders and 'val' in folders:
            train_folder = os.path.join(self._json_dir, 'train/')
            train_json_names = [train_sample_name + '.json' \
                                for train_sample_name in os.listdir(train_folder) \
                                if os.path.isdir(os.path.join(train_folder, train_sample_name))]
            
            val_folder = os.path.join(self._json_dir, 'val/')
            val_json_names = [val_sample_name + '.json' \
                              for val_sample_name in os.listdir(val_folder) \
                            if os.path.isdir(os.path.join(val_folder, val_sample_name))]
            
            return train_json_names, val_json_names
        
        train_idxs, val_idxs = train_test_split(range(len(json_names)), 
                                                test_size=val_size)
        train_json_names = [json_names[train_idx] for train_idx in train_idxs]
        val_json_names = [json_names[val_idx] for val_idx in val_idxs]
        
        return train_json_names, val_json_names
    
    def convert(self, val_size=0.2):
        json_names = [file_name for file_name in os.listdir(self._json_dir) \
                      if os.path.isfile(os.path.join(self._json_dir, file_name)) and \
                      file_name.endswith('.json')]
        folders =  [file_name for file_name in os.listdir(self._json_dir) \
                    if os.path.isdir(os.path.join(self._json_dir, file_name))]
        
        train_json_names, val_json_names = self._train_test_split(folders, json_names, val_size)
    
        # convert labelme object to yolo format object, and save them to files
        # also get image from labelme json file and save them under images folder
        for target_dir, json_name in zip(('train/', 'val/'), (train_json_names, val_json_names)):
            json_path = os.path.join(self._json_dir, json_name)
            json_data = json.load(open(json_path))
                
            img_path = self._save_yolo_image(json_data, json_name, target_dir)
                
            yolo_obj_list = self._get_yolo_object_list(json_data, img_path)
            self._save_yolo_label(json_path, target_dir, yolo_obj_list)
    
    def _get_yolo_object_list(self, json_data, img_path):
        def __get_object_desc(obj_port_list):
            __get_dist = lambda int_list: max(int_list) - min(int_list)
            
            x_lists = [port[0] for port in obj_port_list]        
            y_lists = [port[1] for port in obj_port_list]
            
            return min(x_lists), __get_dist(x_lists), min(y_lists), __get_dist(y_lists)
    
        yolo_obj_list = []
        
        for shape in json_data['shapes']:
            obj_x_min, obj_w, obj_y_min, obj_h = __get_object_desc(shape['points'])
                    
            img_h, img_w, _ = cv2.imread(img_path).shape
                    
            yolo_center_x= round(float((obj_x_min + obj_w / 2.0) / img_w), 6)
            yolo_center_y = round(float((obj_y_min + obj_h / 2.0) / img_h), 6)
            yolo__w = round(float(obj_w / img_w), 6)
            yolo_h = round(float(obj_h / img_h), 6)
            
            label_id = self._label_id_map[shape['label']]
            
            yolo_obj_list.append((label_id, yolo_center_x, yolo_center_y, yolo__w, yolo_h))
            
        return yolo_obj_list
    
    def _save_yolo_label(self, json_path, yolo_obj_list):
        txt_path = json_path.replace('.json', '.text')

        with open(txt_path, 'w+') as f:
            for yolo_obj in yolo_obj_list:
                f.write('%s %s %s %s %s\n' % yolo_obj)
    
    def _save_yolo_image(self, json_data, json_name, target_dir):
        img = utils.img_b64_to_arr(json_data['imageData'])
        
        img_name = json_name.replace('.json', '.png')
        img_path = os.path.join(self._image_dir_path, target_dir,img_name )
        
        PIL.Image.fromarray(img).save(img_path)
        
        return img_path

if __name__ == '__main__':
    argv = sys.argv[1:]
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_dir',type=str,
                        help='Please input the path of the labelme json files.')
    parser.add_argument('--val_size',type=str,
                        help='Please input the validation dataset size, for example 0.1 ')
    json_dir, val_size = parser.parse_args(argv)
    
    convertor = Labelme2YOLO(json_dir)
    convertor.convert(val_size=val_size)
    
