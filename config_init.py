#!/usr/bin/python
#-*-encoding:utf-8-*-
import csv
import platform
import time
import numpy as np
from yaml import load
from find_lane import DashboardCamera, LaneFinder
from fuzzy_convert import FuzzyConvert
from pid_control import PIDControl
import os
import tensorflow as tf
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
from tensorflow.python.platform import tf_logging as logging
from tensorflow.contrib.layers.python.layers import initializers
class Client:
    def __init__(self, unity3d_simulation_status):
        #config info
        config_path = os.path.join(os.path.dirname(__file__), '../cfg_file/config.yaml')
        if unity3d_simulation_status == True:
            config_path = os.path.join(os.path.dirname(__file__), '../cfg_file/config_unity3d_sim.yaml')
        with open(config_path,'rb') as f:
            cont = f.read()
        cf = load(cont)

        #1, calibration config
        camera_ost_path = os.path.join(os.path.dirname(__file__), cf.get('camera')['front_center']['ost_path'])
        with open(camera_ost_path,'rb') as camera_f:
            camera_cont = camera_f.read()
        camera_cf = load(camera_cont)
 
        camera_matrix=camera_cf.get('camera_matrix')['data']
        rows=camera_cf.get('camera_matrix')['rows']
        cols=camera_cf.get('camera_matrix')['cols']
        self.camera_matrix=np.reshape(camera_matrix, (rows, cols))

        distortion_coefficients=camera_cf.get('distortion_coefficients')['data']
        rows=camera_cf.get('distortion_coefficients')['rows']
        cols=camera_cf.get('distortion_coefficients')['cols']
        self.distortion_coefficients=np.reshape(distortion_coefficients, (rows, cols))


        #2, vision config
        self.img_h = cf.get('vision')['img_h']
        self.img_w = cf.get('vision')['img_w']
        scale_y_m_per_pix=cf.get('vision')['scale']['y_dis']/cf.get('vision')['scale']['y_pix']
        scale_x_m_per_pix=cf.get('vision')['scale']['x_dis']/cf.get('vision')['scale']['x_pix']
        scale= (scale_y_m_per_pix,scale_x_m_per_pix)


        img_source_top=cf.get('vision')['img_source']['top_line']
        img_source_bottom=cf.get('vision')['img_source']['bottom_line']
        img_source_height=cf.get('vision')['img_source']['height']
        img_source=np.float32([((self.img_w-img_source_top)/2, self.img_h-img_source_height), ((self.img_w+img_source_top)/2, self.img_h-img_source_height),((self.img_w-img_source_bottom)/2, self.img_h), ((self.img_w+img_source_bottom)/2,self.img_h)])


        img_destination_w=cf.get('vision')['img_destination']['width']
        img_destination_h=cf.get('vision')['img_destination']['height']
        img_destination=np.float32([((self.img_w-img_destination_w)/2, self.img_h-img_destination_h), ((self.img_w+img_destination_w)/2, self.img_h-img_destination_h),((self.img_w-img_destination_w)/2, self.img_h), ((self.img_w+img_destination_w)/2,self.img_h)])


        self.camera = DashboardCamera(scale_correction=scale, h=self.img_h, w=self.img_w,source = img_source,destination = img_destination)
        
        window_shape_h = cf.get('vision')['lane_finder']['window_shape_h']
        window_shape_w = cf.get('vision')['lane_finder']['window_shape_w']
        search_margin = cf.get('vision')['lane_finder']['search_margin']
        max_frozen_dur = cf.get('vision')['lane_finder']['max_frozen_dur']
        self.lane_finder = LaneFinder( self.camera, (window_shape_h,window_shape_w), search_margin, max_frozen_dur)

        self.bound_point_num = cf.get('vision')['bound_point_num']


        ckpt=tf.train.get_checkpoint_state(os.path.join(os.path.dirname(__file__), cf.get('vision')['model']['checkpoint_state']))
        self.saver = tf.train.import_meta_graph(os.path.join(os.path.dirname(__file__), cf.get('vision')['model']['meta_graph']))

        self.graph = tf.get_default_graph()
        self.input_tensor = self.graph.get_tensor_by_name(cf.get('vision')['model']['input_tensor'])
        self.output_tensor = self.graph.get_tensor_by_name(cf.get('vision')['model']['output_tensor'])
        self.session = tf.InteractiveSession()
        self.saver.restore(self.session, ckpt.model_checkpoint_path)   



        #3, fuzzy convert config
        angle_step = cf.get('fuzzy_convert')['angle_step']
        distance_step = cf.get('fuzzy_convert')['distance_step']
        cte_step = cf.get('fuzzy_convert')['cte_step']
        self.fuzzyconvert=FuzzyConvert(angle_axis_step = angle_step , distance_axis_step = distance_step, cte_axis_step = cte_step)



        #4, control config
        kp=cf.get('control')['kp']
        ki=cf.get('control')['ki']
        kd=cf.get('control')['kd']
        self.limited_speed=cf.get('control')['limited_speed']
        self.throttle=cf.get('control')['throttle']
        self.pid=PIDControl(kp,ki,kd)
        self.gear=cf.get('control')['original_gear']

 
        print ('...waiting for message..' )
        #5 log info during driving
        #log_file_name =  cf.get('log_file')['dir_path']+str(time.time())+'.log'
        # csvfile = open(log_file_name, 'w', newline='')
        # self.writer = csv.writer(csvfile)
        # self.writer.writerow(['time', 'shift', 'speed', 'steering angle', 'derivat tan', 'derivat angle', 'x', 'cte', 'recv_shift', 'recv_speed','recv_steering'])

        #6 replay dataset

        replay_file_path = os.path.join(os.path.dirname(__file__), cf.get('replay')['file_path'])
        self.tags_list=[]
        self.replay_list=[]
        temp_list=[]
        with open(replay_file_path) as f:
            f.readline()
            csv_reader = csv.reader(f)
            for row in csv_reader:
                if len(self.tags_list)==0 or self.tags_list[-1]!=row[0]:
                    self.tags_list.append(row[0])
                temp_list.append(row)
            for i in range(len(self.tags_list)):
                current_group=dict()
                current_group['tag']=self.tags_list[i]
                current_replay=[]
                for j in range(len(temp_list)):
                    if self.tags_list[i]==temp_list[j][0]:
                        current_unit=dict()
                        current_unit['shift']=temp_list[j][1]
                        current_unit['speed']=float(temp_list[j][2])
                        current_unit['angle']=float(temp_list[j][3])
                        current_unit['delay']=float(temp_list[j][4])       
                        current_replay.append(current_unit)
                    elif len(current_replay)>0:

                        break
                current_group['replay']=current_replay
                self.replay_list.append(current_group)
        for i in range(len(self.tags_list)):        
            print (self.tags_list[i])
        for i in range(len(self.replay_list)):
            print (self.replay_list[i])
