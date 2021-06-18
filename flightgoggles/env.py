#!/usr/bin/env python
# coding: utf-8

import signal
import os, sys, time, copy, argparse, yaml
import numpy as np
import pandas as pd
import cv2
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image

from flightgoggles_client import FlightGogglesClient as fg_client

import plotly.io as pio
pio.renderers.default = "notebook"

from .utils import *
from .model import *

import functools, traceback
def catch_exception(f):
    @functools.wraps(f)
    def func(self, *args, **kwargs):
        try:
            return f(self, *args, **kwargs)
        except Exception as e:
            print('Caught an exception in {}'.format(f.__name__))
            traceback.print_exc()
            if hasattr(self, 'fg_renderer'):
                self.close()
    return func

def for_all_methods(decorator):
    def decorate(cls):
        for attr in cls.__dict__:
            if callable(getattr(cls, attr)) and attr != '__timeout_handler__':
                setattr(cls, attr, decorator(getattr(cls, attr)))
        return cls
    return decorate

@for_all_methods(catch_exception)
class flightgoggles_env():
    def __init__(self, *args, **kwargs):
        if 'cfg_dir' in kwargs:
            cfg_dir = kwargs['cfg_dir']
        else:
            curr_path = os.path.dirname(os.path.abspath(__file__))
            cfg_dir = curr_path+"/../config/"
        
        if 'cfg_fgclient' in kwargs:
            cfg_fgclient = kwargs['cfg_fgclient']
        else:
            cfg_fgclient = "FlightGogglesClient.yaml"
        
        if 'cfg_uav' in kwargs:
            cfg_uav = kwargs['cfg_uav']
        else:
            cfg_uav = "multicopterDynamicsSim.yaml"
        
        if 'cfg_car' in kwargs:
            cfg_car = kwargs['cfg_car']
        else:
            cfg_car = "carDynamicsSim.yaml"
        
        if 'tmp_dir' in kwargs:
            self.tmp_dir = kwargs['tmp_dir']
        else:
            curr_path = os.path.dirname(os.path.abspath(__file__))
            self.tmp_dir = curr_path+"/../tmp/"
            if not os.path.exists(self.tmp_dir):
                os.makedirs(self.tmp_dir)
            
        fgc_cfg_path = os.path.join(cfg_dir, cfg_fgclient)
        uav_sim_cfg_path_default = os.path.join(cfg_dir, cfg_uav)
        car_sim_cfg_path_default = os.path.join(cfg_dir, cfg_car)

        self.fg_renderer = dict()
        self.fg_render_cam_info = dict()
        self.fg_render_obj_info = dict()
        self.vehicle_set = dict()
        self.camera_set = dict()
        self.object_set = dict()
        self.static_camera_keys = []
        self.static_camera_time = 0
        
        with open(fgc_cfg_path, 'r') as stream:
            try:
                cfg = yaml.safe_load(stream)
                if 'state' in cfg:
                    self.cam_width = cfg['state']['camWidth']
                    self.cam_height = cfg['state']['camHeight']
                
                if 'renderer' in cfg:
                    for renderer_key in cfg['renderer'].keys():
                        self.fg_renderer[renderer_key] = \
                            fg_client(
                                yaml_path = fgc_cfg_path,
                                input_port = cfg['renderer'][renderer_key]["inputPort"],
                                output_port = cfg['renderer'][renderer_key]["outputPort"])
                        self.fg_render_cam_info[renderer_key] = 0
                        self.fg_render_obj_info[renderer_key] = 0
                
                if 'camera_model' in cfg:
                    ts = time.strftime('%Y-%b-%d-%H:%M:%S', time.gmtime())
                    self.camera_img_dir = os.path.join(self.tmp_dir, ts)
                    if not os.path.exists(self.camera_img_dir):
                        os.makedirs(self.camera_img_dir)
                    for camera_key in cfg['camera_model'].keys():
                        self.camera_set[cfg['camera_model'][camera_key]['ID']] = dict()
                        renderer_key = cfg['camera_model'][camera_key]['renderer']
                        # self.camera_set[cfg['camera_model'][camera_key]['ID']]["index"] = int(camera_key)
                        # self.camera_set[cfg['camera_model'][camera_key]['ID']]["index"] = np.int(cfg['camera_model'][camera_key]['outputIndex'])
                        self.camera_set[cfg['camera_model'][camera_key]['ID']]["index"] = self.fg_render_cam_info[renderer_key]
                        self.fg_render_cam_info[renderer_key] += 1
                        self.camera_set[cfg['camera_model'][camera_key]['ID']]["channels"] = cfg['camera_model'][camera_key]['channels']
                        self.camera_set[cfg['camera_model'][camera_key]['ID']]["freq"] = cfg['camera_model'][camera_key]['freq']
                        self.camera_set[cfg['camera_model'][camera_key]['ID']]["renderer"] = renderer_key
                        if 'initialPose' in cfg['camera_model'][camera_key]:
                            self.camera_set[cfg['camera_model'][camera_key]['ID']]["initialPose"] = cfg['camera_model'][camera_key]['initialPose']
                            self.camera_set[cfg['camera_model'][camera_key]['ID']]["currentPos"] = cfg['camera_model'][camera_key]['initialPose'][:3]
                            self.camera_set[cfg['camera_model'][camera_key]['ID']]["currentAtt"] = cfg['camera_model'][camera_key]['initialPose'][3:]
                        else:
                            initialPose_t = np.array([0,0,0,1,0,0,0])
                            self.camera_set[cfg['camera_model'][camera_key]['ID']]["initialPose"] = initialPose_t
                            self.camera_set[cfg['camera_model'][camera_key]['ID']]["currentPos"] = initialPose_t[:3]
                            self.camera_set[cfg['camera_model'][camera_key]['ID']]["currentAtt"] = initialPose_t[3:]
                        self.camera_set[cfg['camera_model'][camera_key]['ID']]["logs"] = []
                        if 'hasCollisionCheck' in cfg['camera_model'][camera_key].keys():
                            hasCollisionCheck_t = cfg['camera_model'][camera_key]['hasCollisionCheck']
                        else:
                            hasCollisionCheck_t = False
                        self.fg_renderer[renderer_key].addCamera(
                            cfg['camera_model'][camera_key]['ID'], 
                            np.int(self.camera_set[cfg['camera_model'][camera_key]['ID']]["index"]),
                            np.int(cfg['camera_model'][camera_key]['outputShaderType']),
                            hasCollisionCheck_t)
                        self.static_camera_keys.append(cfg['camera_model'][camera_key]['ID'])
                        img_dir_t = os.path.join(self.camera_img_dir, cfg['camera_model'][camera_key]['ID'])
                        if not os.path.exists(img_dir_t):
                            os.makedirs(img_dir_t)
                        self.camera_set[cfg['camera_model'][camera_key]['ID']]["img_dir"] = img_dir_t

                if 'objects' in cfg:
                    for object_key in cfg['objects'].keys():
                        object_id = cfg['objects'][object_key]['ID']
                        object_prefab_id = cfg['objects'][object_key]['prefabID']
                        object_size_x = cfg['objects'][object_key]['size_x']
                        object_size_y = cfg['objects'][object_key]['size_y']
                        object_size_z = cfg['objects'][object_key]['size_z']
                        if 'renderer' in cfg['objects'][object_key]:
                            renderer_key_t = cfg['objects'][object_key]['renderer']
                        else:
                            renderer_key_t = -1

                        self.object_set[cfg['objects'][object_key]['ID']] = dict()
                        self.object_set[cfg['objects'][object_key]['ID']]['renderer'] = renderer_key_t
                        # self.object_set[cfg['objects'][object_key]['ID']]['index'] = np.int(cfg['objects'][object_key]['outputIndex'])
                        self.object_set[cfg['objects'][object_key]['ID']]['index'] = dict()

                        if 'initialPose' in cfg['objects'][object_key].keys():
                            obj_pos_t = cfg['objects'][object_key]['initialPose'][:3]
                            obj_att_t = cfg['objects'][object_key]['initialPose'][3:]
                        else:
                            initialPose_t = np.array([0,0,0,1,0,0,0])
                            obj_pos_t = initialPose_t[:3]
                            obj_att_t = initialPose_t[3:]
                        
                        if renderer_key_t == -1:
                            for renderer_key in self.fg_renderer.keys():
                                self.fg_renderer[renderer_key].addObject(
                                    object_id, object_prefab_id, 
                                    np.double(object_size_x), 
                                    np.double(object_size_y), 
                                    np.double(object_size_z))
                                self.object_set[cfg['objects'][object_key]['ID']]['index'][renderer_key] = self.fg_render_obj_info[renderer_key]
                                self.fg_renderer[renderer_key] \
                                    .setObjectPose(obj_pos_t, obj_att_t, self.fg_render_obj_info[renderer_key])
                                self.fg_render_obj_info[renderer_key] += 1
                        else:
                            self.fg_renderer[renderer_key_t].addObject(
                                object_id, object_prefab_id, 
                                np.double(object_size_x), 
                                np.double(object_size_y), 
                                np.double(object_size_z))
                            self.object_set[cfg['objects'][object_key]['ID']]['index'][renderer_key_t] = self.fg_render_obj_info[renderer_key_t]
                            self.fg_renderer[renderer_key_t] \
                                .setObjectPose(obj_pos_t, obj_att_t, self.fg_render_obj_info[renderer_key_t])
                            self.fg_render_obj_info[renderer_key_t] += 1

                if 'vehicle_model' in cfg:
                    for vehicle_key in cfg['vehicle_model'].keys():
                        if 'cameraInfo' in cfg['vehicle_model'][vehicle_key]:
                            camera_info_t = cfg['vehicle_model'][vehicle_key]['cameraInfo']
                        else:
                            camera_info_t = dict()
                            
                        if 'objectsInfo' in cfg['vehicle_model'][vehicle_key]:
                            objects_info_t = cfg['vehicle_model'][vehicle_key]['objectsInfo']
                        else:
                            objects_info_t = dict()
                        
                        for camera_id in camera_info_t.keys():
                            if camera_id not in self.camera_set.keys():
                                print("Camera key {} is not registered on camera_model".format(camera_id))
                            camera_info_t[camera_id]["freq"] = self.camera_set[camera_id]['freq']
                            if camera_id in self.static_camera_keys:
                                self.static_camera_keys.remove(camera_id)
                        if cfg['vehicle_model'][vehicle_key]['type'] == "uav":
                            if 'config_filename' in cfg['vehicle_model'][vehicle_key]:
                                cfg_uav_t = cfg['vehicle_model'][vehicle_key]['config_filename']
                                if os.path.exist(cfg_uav_t):
                                    uav_sim_cfg_path = cfg_uav_t
                                elif os.path.exist(os.path.join(cfg_dir, cfg_uav_t)):
                                    uav_sim_cfg_path = os.path.join(cfg_dir, cfg_uav_t)
                                else:
                                    uav_sim_cfg_path = uav_sim_cfg_path_default
                            else:
                                uav_sim_cfg_path = uav_sim_cfg_path_default
                            vehicle_tmp = MulticopterModel(
                                cfg_path=uav_sim_cfg_path,
                                id=vehicle_key,
                                init_pose=np.array(cfg['vehicle_model'][vehicle_key]['initialPose']),
                                camera_info=camera_info_t,
                                objects_info=objects_info_t,
                                imu_freq=cfg['vehicle_model'][vehicle_key]['imu_freq'])
                        elif cfg['vehicle_model'][vehicle_key]['type'] == "car":
                            if 'config_filename' in cfg['vehicle_model'][vehicle_key]:
                                cfg_car_t = cfg['vehicle_model'][vehicle_key]['config_filename']
                                if os.path.exist(cfg_car_t):
                                    car_sim_cfg_path = cfg_car_t
                                elif os.path.exist(os.path.join(cfg_dir, cfg_car_t)):
                                    car_sim_cfg_path = os.path.join(cfg_dir, cfg_car_t)
                                else:
                                    car_sim_cfg_path = car_sim_cfg_path_default
                            else:
                                car_sim_cfg_path = car_sim_cfg_path_default
                            vehicle_tmp = CarModel(
                                cfg_path=car_sim_cfg_path,
                                id=vehicle_key,
                                init_pose=np.array(cfg['vehicle_model'][vehicle_key]['initialPose']),
                                camera_info=camera_info_t,
                                objects_info=objects_info_t,
                                imu_freq=cfg['vehicle_model'][vehicle_key]['imu_freq'])
                        else:
                            continue
                        self.vehicle_set[vehicle_key] = dict()
                        self.vehicle_set[vehicle_key]["type"] = cfg['vehicle_model'][vehicle_key]['type']
                        self.vehicle_set[vehicle_key]["model"] = vehicle_tmp
                        if 'objectsInfo' in cfg['vehicle_model'][vehicle_key]:
                            self.vehicle_set[vehicle_key]["objectsInfo"] = cfg['vehicle_model'][vehicle_key]['objectsInfo']
                        self.vehicle_set[vehicle_key]["logs"] = []
                    
                    for static_camera_id in self.static_camera_keys:
                        self.camera_set[static_camera_id]["cam_time_last"] = 0
              
                if 'map' in cfg:  
                    self.map_x_min = cfg['state']['map']['x_min']
                    self.map_x_max = cfg['state']['map']['x_max']
                    self.map_y_min = cfg['state']['map']['y_min']
                    self.map_y_max = cfg['state']['map']['y_max']
                    self.map_margin = cfg['state']['map']['map_margin']
                    self.map_scale = cfg['state']['map']['map_scale']
                    curr_path = os.path.dirname(os.path.abspath(__file__))
                    self.map_filepath = os.path.join(curr_path, "../res/", cfg['state']['map']['filename'])
                else:
                    self.map_filepath = ""
                
            except yaml.YAMLError as exc:
                print(exc)
        
        self.initialize_state()
        return
    
    def __del__(self):
        self.close()
        return
    
    def close(self):
        if hasattr(self, 'fg_renderer'):
            d = dict.fromkeys(self.fg_renderer.keys(),[])
            for key in d.keys():
                self.fg_renderer[key].terminate()
                del(self.fg_renderer[key])
        return
    
    def manual_seed(self, seed):
        for vehicle_key in self.vehicle_set.keys():
            if self.vehicle_set[vehicle_key]["type"] == "car":
                self.vehicle_set[vehicle_key]["model"].setNoiseSeed(seed)
            elif self.vehicle_set[vehicle_key]["type"] == "drone":
                self.vehicle_set[vehicle_key]["model"].setRandomSeed(seed, seed)
        return

    def set_state_vehicle(self, vehicle_id, **kwargs):
        self.vehicle_set[vehicle_id]["model"].set_state(**kwargs)
        
        sim_time_last_max = 0
        for vehicle_key in self.vehicle_set.keys():
            sim_time_last = self.vehicle_set[vehicle_key]["model"].sim_time
            if sim_time_last_max == 0 or sim_time_last_max < sim_time_last:
                sim_time_last_max = sim_time_last
        if sim_time_last_max == 0:
            self.reset_state()
        return
    
    def get_state(self, vehicle_id):
        return self.vehicle_set[vehicle_id]["model"].get_state()

    def set_state_object(self, object_id, position, attitude):
        if self.object_set[object_id]["renderer"] == -1:
            for renderer_key in self.fg_renderer.keys():
                self.fg_renderer[renderer_key] \
                     .setObjectPose(position, attitude, self.object_set[object_id]["index"][renderer_key])
        else:
            self.fg_renderer[self.object_set[object_id]["renderer"]] \
                 .setObjectPose(position, attitude, self.object_set[object_id]["index"][self.object_set[object_id]["renderer"]])
        
        return
    
    def set_state_camera(self, camera_id, position, attitude, flag_save_logs=False, flag_update_simulation=True):
        self.fg_renderer[self.camera_set[camera_id]["renderer"]] \
            .setCameraPose(position, attitude, self.camera_set[camera_id]["index"])
        self.fg_renderer[self.camera_set[camera_id]["renderer"]] \
            .setStateTime(self.fg_renderer[self.camera_set[camera_id]["renderer"]].getTimestamp())
        self.fg_renderer[self.camera_set[camera_id]["renderer"]].requestRender()

        if flag_save_logs:
            res, res_collision, res_landmark, res_lidar = self.fg_renderer[self.camera_set[camera_id]["renderer"]].getImage()
            res_t = dict()
            for res_key in res.keys():
                if res_key == camera_id:
                    img = res[res_key]
                    img_reshape = np.reshape(img,(self.cam_height,self.cam_width,self.camera_set[res_key]["channels"]))
                    # img_reshape = cv2.flip(cv2.cvtColor(img_reshape, cv2.COLOR_BGR2RGB),-1)
                    #img_reshape = cv2.cvtColor(img_reshape, cv2.COLOR_BGR2RGB)
                    res_t[res_key] = img_reshape
                    img_log = dict()
                    img_log["timestamp"] = 0.0
                    
                    file_path = os.path.join(self.camera_set[res_key]["img_dir"],"{0:019d}.png".format(np.uint64(img_log["timestamp"]*1e9)))
                    if (self.camera_set[res_key]["channels"] == 1):
                        cv2.imwrite(file_path, img_reshape.astype(np.uint16))
                    else:
                        cv2.imwrite(file_path, img_reshape.astype(np.uint8))
                    
                    img_log["data"] = file_path
                    self.camera_set[res_key]["logs"].append(img_log)
                    
        if flag_update_simulation:
            sim_time_last_max = 0
            for vehicle_key in self.vehicle_set.keys():
                sim_time_last = self.vehicle_set[vehicle_key]["model"].sim_time
                if sim_time_last_max == 0 or sim_time_last_max < sim_time_last:
                    sim_time_last_max = sim_time_last
            if sim_time_last_max == 0:
                self.reset_state()
        return
    
    def get_camera_image(self, camera_id):
        return self.camera_set[camera_id]["logs"]

    def initialize_state(self):
        for camera_key in self.camera_set.keys():
            self.camera_set[camera_key]["logs"] = []
            position = self.camera_set[camera_key]['initialPose'][:3]
            attitude = self.camera_set[camera_key]['initialPose'][3:]
            self.fg_renderer[self.camera_set[camera_key]["renderer"]].setCameraPose(position, attitude, self.camera_set[camera_key]["index"])
            self.camera_set[camera_key]["currentPos"] = position
            self.camera_set[camera_key]["currentAtt"] = attitude
        
        for vehicle_key in self.vehicle_set.keys():
            self.vehicle_set[vehicle_key]["model"].initialize_state()
            self.vehicle_set[vehicle_key]["logs"] = [copy.deepcopy(self.vehicle_set[vehicle_key]["model"].get_state())]
        
        self._update_state_camera()
        self.flag_started = False
        return

    def reset_state(self):
        for vehicle_key in self.vehicle_set.keys():
            self.vehicle_set[vehicle_key]["model"].reset_state()
            self._update_state(vehicle_key)
            self.vehicle_set[vehicle_key]["logs"] = [copy.deepcopy(self.vehicle_set[vehicle_key]["model"].get_state())]
        
        if len(self.camera_set.keys()) > 0:
            ts = time.strftime('%Y-%b-%d-%H:%M:%S', time.gmtime())
            self.camera_img_dir = os.path.join(self.tmp_dir, ts)
        
        for camera_key in self.camera_set.keys():
            img_dir_t = os.path.join(self.camera_img_dir, camera_key)
            if not os.path.exists(img_dir_t):
                os.makedirs(img_dir_t)
            self.camera_set[camera_key]["img_dir"] = img_dir_t

            if not os.path.exists(self.camera_img_dir):
                os.makedirs(self.camera_img_dir)
            if len(self.camera_set[camera_key]["logs"]) != 0:
                if self.flag_started:
                    self.camera_set[camera_key]["logs"] = [self.camera_set[camera_key]["logs"][-1]]
                else:
                    self.camera_set[camera_key]["logs"] = []
        if not self.flag_started:
            self._update_state_camera()
        return
    
    def proceed(self, vehicle_id, speed_command, steering_angle_command, duration):
        if self.vehicle_set[vehicle_id]["type"] != "car":
            return
        self.vehicle_set[vehicle_id]["model"].proceed(speed_command, steering_angle_command, duration)
        self._update_state(vehicle_id)
        self.flag_started = True
        return

    def proceed_motor_speed(self, vehicle_id, motor_command, duration):
        if self.vehicle_set[vehicle_id]["type"] != "uav":
            return
        self.vehicle_set[vehicle_id]["model"].proceed_motor_speed(motor_command, duration)
        self._update_state(vehicle_id)
        self.flag_started = True
        return

    def proceed_angular_rate(self, vehicle_id, angular_rate_command, thrust_command, duration):
        if self.vehicle_set[vehicle_id]["type"] != "uav":
            return
        self.vehicle_set[vehicle_id]["model"].proceed_angular_rate(angular_rate_command, thrust_command, duration)
        self._update_state(vehicle_id)
        self.flag_started = True
        return
    
    def proceed_waypoint(self, vehicle_id, waypoint_command, duration):
        if self.vehicle_set[vehicle_id]["type"] != "uav":
            return
        self.vehicle_set[vehicle_id]["model"].proceed_waypoint(waypoint_command, duration)
        self._update_state(vehicle_id)
        self.flag_started = True
        return

    def save_logs(self, vehicle_id=None, save_dir="data/"):
        # Save IMU and Camera data
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        vehicle_set = []
        if np.all(vehicle_id == None):
            vehicle_set = self.vehicle_set.keys()
        else:
            vehicle_set = [vehicle_id]
        
        for vehicle_id in vehicle_set:
            if self.vehicle_set[vehicle_id]["type"] == "uav":
                arr_timestamp = []
                arr_acc_raw_x = []
                arr_acc_raw_y = []
                arr_acc_raw_z = []
                arr_gyro_raw_x = []
                arr_gyro_raw_y = []
                arr_gyro_raw_z = []
                arr_acc_x = []
                arr_acc_y = []
                arr_acc_z = []
                arr_gyro_x = []
                arr_gyro_y = []
                arr_gyro_z = []
                for log in self.vehicle_set[vehicle_id]["logs"]:
                    arr_timestamp.append(np.uint64(log["timestamp"]*1e9))
                    arr_acc_raw_x.append(log["acceleration_raw"][0])
                    arr_acc_raw_y.append(log["acceleration_raw"][1])
                    arr_acc_raw_z.append(log["acceleration_raw"][2])
                    arr_gyro_raw_x.append(log["gyroscope_raw"][0])
                    arr_gyro_raw_y.append(log["gyroscope_raw"][1])
                    arr_gyro_raw_z.append(log["gyroscope_raw"][2])
                    arr_acc_x.append(log["acceleration"][0])
                    arr_acc_y.append(log["acceleration"][1])
                    arr_acc_z.append(log["acceleration"][2])
                    arr_gyro_x.append(log["angular_velocity"][0])
                    arr_gyro_y.append(log["angular_velocity"][1])
                    arr_gyro_z.append(log["angular_velocity"][2])
                df = pd.DataFrame({
                    "timestamp":arr_timestamp,
                    "gyroscope.x":arr_gyro_raw_x,
                    "gyroscope.y":arr_gyro_raw_y,
                    "gyroscope.z":arr_gyro_raw_z,
                    "acceleration.x":arr_acc_raw_x,
                    "acceleration.y":arr_acc_raw_y,
                    "acceleration.z":arr_acc_raw_z,
                    "gyroscope_filtered.x":arr_gyro_x,
                    "gyroscope_filtered.y":arr_gyro_y,
                    "gyroscope_filtered.z":arr_gyro_z,
                    "acceleration_filtered.x":arr_acc_x,
                    "acceleration_filtered.y":arr_acc_y,
                    "acceleration_filtered.z":arr_acc_z})
                df.to_csv(os.path.join(save_dir,"{}_imu.csv".format(vehicle_id)), sep=',', index=False)

                arr_timestamp = []
                arr_pos_x = []
                arr_pos_y = []
                arr_pos_z = []
                arr_vel_x = []
                arr_vel_y = []
                arr_vel_z = []
                arr_att_x = []
                arr_att_y = []
                arr_att_z = []
                arr_ms_0 = []
                arr_ms_1 = []
                arr_ms_2 = []
                arr_ms_3 = []
                for log in self.vehicle_set[vehicle_id]["logs"]:
                    arr_timestamp.append(np.uint64(log["timestamp"]*1e9))
                    arr_pos_x.append(log["position"][0])
                    arr_pos_y.append(log["position"][1])
                    arr_pos_z.append(log["position"][2])
                    arr_vel_x.append(log["velocity"][0])
                    arr_vel_y.append(log["velocity"][1])
                    arr_vel_z.append(log["velocity"][2])
                    arr_att_x.append(log["attitude_euler_angle"][0])
                    arr_att_y.append(log["attitude_euler_angle"][1])
                    arr_att_z.append(log["attitude_euler_angle"][2])
                    arr_ms_0.append(log["motor_speed"][0])
                    arr_ms_1.append(log["motor_speed"][1])
                    arr_ms_2.append(log["motor_speed"][2])
                    arr_ms_3.append(log["motor_speed"][3])
                df = pd.DataFrame({
                    "timestamp":arr_timestamp,
                    "position.x":arr_pos_x,
                    "position.y":arr_pos_y,
                    "position.z":arr_pos_z,
                    "velocity.x":arr_vel_x,
                    "velocity.y":arr_vel_y,
                    "velocity.z":arr_vel_z,
                    "attitude.x":arr_att_x,
                    "attitude.y":arr_att_y,
                    "attitude.z":arr_att_z,
                    "motor_speed.0":arr_ms_0,
                    "motor_speed.1":arr_ms_1,
                    "motor_speed.2":arr_ms_2,
                    "motor_speed.3":arr_ms_3})
                df.to_csv(os.path.join(save_dir,"{}_sim.csv".format(vehicle_id)), sep=',', index=False)
            if self.vehicle_set[vehicle_id]["type"] == "car":
                arr_timestamp = []
                arr_pos_x = []
                arr_pos_y = []
                arr_pos_z = []
                arr_vel_x = []
                arr_vel_y = []
                arr_vel_z = []
                arr_att = []
                for log in self.vehicle_set[vehicle_id]["logs"]:
                    arr_timestamp.append(np.uint64(log["timestamp"]*1e9))
                    arr_pos_x.append(log["position"][0])
                    arr_pos_y.append(log["position"][1])
                    arr_pos_z.append(log["position"][2])
                    arr_vel_x.append(log["velocity"][0])
                    arr_vel_y.append(log["velocity"][1])
                    arr_vel_z.append(log["velocity"][2])
                    arr_att.append(log["heading"])
                df = pd.DataFrame({
                    "timestamp":arr_timestamp,
                    "position.x":arr_pos_x,
                    "position.y":arr_pos_y,
                    "position.z":arr_pos_z,
                    "velocity.x":arr_vel_x,
                    "velocity.y":arr_vel_y,
                    "velocity.z":arr_vel_z,
                    "heading":arr_att})
                df.to_csv(os.path.join(save_dir,"{}_sim.csv".format(vehicle_id)), sep=',', index=False)
        
        for camera_id in self.camera_set.keys():
            dir_camera = os.path.join(save_dir,"{}/".format(camera_id))
            if not os.path.exists(dir_camera):
                os.makedirs(dir_camera)
            for log in self.camera_set[camera_id]["logs"]:
                file_path = os.path.join(dir_camera,"{0:019d}.png".format(np.uint64(log["timestamp"]*1e9)))
                img_t = cv2.imread(log["data"], -1)
                cv2.imwrite(file_path, img_t)
        return

    def _update_state(self, vehicle_id):
        # self.vehicle_set[vehicle_id]["model"].update_state_vehicle(duration)
        # self.vehicle_set[vehicle_id]["model"].update_state_imu(duration)
        self.vehicle_set[vehicle_id]["logs"].append(copy.deepcopy(self.vehicle_set[vehicle_id]["model"].get_state()))
        
        # update object position
        if 'objectsInfo' in self.vehicle_set[vehicle_id]:
            for objects_key in self.vehicle_set[vehicle_id]['objectsInfo'].keys():
                self.set_state_object(
                    objects_key,
                    self.vehicle_set[vehicle_id]["model"].objects_pose[objects_key]["position"], 
                    self.vehicle_set[vehicle_id]["model"].objects_pose[objects_key]["attitude"])
        
        for cam_key in self.vehicle_set[vehicle_id]["model"].camera_info.keys():
            if self.vehicle_set[vehicle_id]["model"].camera_pose[cam_key]["flag_update"]:
                self._update_state_camera_single(cam_key, 
                    self.vehicle_set[vehicle_id]["model"].camera_pose[cam_key]["position"], 
                    self.vehicle_set[vehicle_id]["model"].camera_pose[cam_key]["attitude"],
                    self.vehicle_set[vehicle_id]["model"].camera_pose[cam_key]["cam_time_last"])
                self.vehicle_set[vehicle_id]["model"].camera_pose[cam_key]["flag_update"] = False
        
        # Update static camera
        cam_time_last_min = -1
        for vehicle_key in self.vehicle_set.keys():
            for cam_key in self.vehicle_set[vehicle_key]["model"].camera_info.keys():
                cam_time_last = self.vehicle_set[vehicle_key]["model"].camera_pose[cam_key]["cam_time_last"]
                if cam_time_last_min == -1 or cam_time_last_min > cam_time_last:
                    cam_time_last_min = cam_time_last
        if cam_time_last_min < 0:
            for vehicle_key in self.vehicle_set.keys():
                cam_time_last = self.vehicle_set[vehicle_key]["model"].sim_time
                if cam_time_last_min == -1 or cam_time_last_min > cam_time_last:
                    cam_time_last_min = cam_time_last
        if self.static_camera_time < cam_time_last_min:
            for cam_key in self.static_camera_keys:
                while cam_time_last_min >= self.camera_set[cam_key]["cam_time_last"]:
                    self._update_state_camera_static(
                        [cam_key],
                        self.camera_set[cam_key]["cam_time_last"])
                    self.camera_set[cam_key]["cam_time_last"] += 1./self.camera_set[cam_key]["freq"]
        return

    def _update_state_camera_single(self, camera_key, position, attitude, cam_time):
        res_t = dict()
        signal.signal(signal.SIGALRM, self.__timeout_handler__)
        while True:
            try:
                try:
                    signal.alarm(1)
                    for key_t in self.camera_set.keys():
                        if key_t != camera_key:
                            # print("key_t: {}".format(self.camera_set[key_t]["index"]))
                            self.fg_renderer[self.camera_set[key_t]["renderer"]].setCameraPose(
                                self.camera_set[key_t]["currentPos"], 
                                self.camera_set[key_t]["currentAtt"], 
                                self.camera_set[key_t]["index"])

                    self.fg_renderer[self.camera_set[camera_key]["renderer"]].setCameraPose(position, attitude, self.camera_set[camera_key]["index"])
                    self.camera_set[camera_key]["currentPos"] = position
                    self.camera_set[camera_key]["currentAtt"] = attitude
                    
                    self.fg_renderer[self.camera_set[camera_key]["renderer"]].setStateTime( \
                        self.fg_renderer[self.camera_set[camera_key]["renderer"]].getTimestamp())
                    self.fg_renderer[self.camera_set[camera_key]["renderer"]].requestRender()
                    res, res_collision, res_landmark, res_lidar = self.fg_renderer[self.camera_set[camera_key]["renderer"]].getImage()
                    for res_key in res.keys():
                        if res_key == camera_key:
                            img = res[res_key]
                            img_reshape = np.reshape(img,(self.cam_height,self.cam_width,self.camera_set[res_key]["channels"]))
                            # img_reshape = cv2.flip(cv2.cvtColor(img_reshape, cv2.COLOR_BGR2RGB),-1)
                            # img_reshape = cv2.cvtColor(img_reshape, cv2.COLOR_BGR2RGB)
                            res_t[res_key] = img_reshape
                            # self.camera_set[res_key]["image"] = img_reshape
                            # print("key {} - len {}".format(res_key, len(self.camera_set[res_key]["logs"])))
                            img_log = dict()
                            img_log["timestamp"] = cam_time
                            
                            file_path = os.path.join(self.camera_set[res_key]["img_dir"],"{0:019d}.png".format(np.uint64(cam_time*1e9)))
                            if (self.camera_set[res_key]["channels"] == 1):
                                cv2.imwrite(file_path, img_reshape.astype(np.uint16))
                            else:
                                cv2.imwrite(file_path, img_reshape.astype(np.uint8))

                            img_log["data"] = file_path
                            self.camera_set[res_key]["logs"].append(img_log)
                except Exception:
                    print('Got exception getting image')
                    continue
            except Exception:
                print('Got exception getting image 2')
                continue
            signal.alarm(0)
            break
        return res_t

    def _update_state_camera_static(self, camera_id_set=[], cam_time=0):
        res_t = dict()
        signal.signal(signal.SIGALRM, self.__timeout_handler__)
        static_camera_keys_t = []
        if len(camera_id_set) == 0:
            static_camera_keys_t = self.static_camera_keys
        else:
            static_camera_keys_t = camera_id_set
            
        while True:
            try:
                try:
                    signal.alarm(1)
                    # signal.setitimer(signal.ITIMER_REAL, TIMEOUT)
                    renderer_key_set = []
                    for vehicle_key in self.vehicle_set.keys():
                        for camera_key in self.vehicle_set[vehicle_key]["model"].camera_info.keys():
                            pos_t = self.vehicle_set[vehicle_key]["model"].camera_pose[camera_key]["position"]
                            att_t = self.vehicle_set[vehicle_key]["model"].camera_pose[camera_key]["attitude"]
                            
                            self.fg_renderer[self.camera_set[camera_key]["renderer"]].setCameraPose( \
                                pos_t, att_t, self.camera_set[camera_key]["index"])
                            self.camera_set[camera_key]["currentPos"] = pos_t
                            self.camera_set[camera_key]["currentAtt"] = att_t
                            if self.camera_set[camera_key]["renderer"] not in renderer_key_set:
                                renderer_key_set.append(self.camera_set[camera_key]["renderer"])
                    for key_t in static_camera_keys_t:
                        self.fg_renderer[self.camera_set[key_t]["renderer"]].setCameraPose(
                            self.camera_set[key_t]["currentPos"], 
                            self.camera_set[key_t]["currentAtt"], 
                            self.camera_set[key_t]["index"])
                        if self.camera_set[key_t]["renderer"] not in renderer_key_set:
                            renderer_key_set.append(self.camera_set[key_t]["renderer"])
                    
                    for renderer_key in renderer_key_set:
                        self.fg_renderer[renderer_key].setStateTime(self.fg_renderer[renderer_key].getTimestamp())
                        self.fg_renderer[renderer_key].requestRender()
                        res, res_collision, res_landmark, res_lidar = self.fg_renderer[renderer_key].getImage()
                        for res_key in res.keys():
                            if res_key in static_camera_keys_t:
                                img = res[res_key]
                                img_reshape = np.reshape(img,(self.cam_height,self.cam_width,self.camera_set[res_key]["channels"]))
                                res_t[res_key] = img_reshape
                                # self.camera_set[res_key]["image"] = img_reshape
                                # img_reshape = cv2.flip(cv2.cvtColor(img_reshape, cv2.COLOR_BGR2RGB),-1)
                                # img_reshape = cv2.cvtColor(img_reshape, cv2.COLOR_BGR2RGB)
                                img_log = dict()
                                img_log["timestamp"] = cam_time
                            
                                file_path = os.path.join(self.camera_set[res_key]["img_dir"],"{0:019d}.png".format(np.uint64(cam_time*1e9)))
                                if (self.camera_set[res_key]["channels"] == 1):
                                    cv2.imwrite(file_path, img_reshape.astype(np.uint16))
                                else:
                                    cv2.imwrite(file_path, img_reshape.astype(np.uint8))
                                   

                                img_log["data"] = file_path
                                self.camera_set[res_key]["logs"].append(img_log)

                except Exception:
                    continue
            except Exception:
                continue
            signal.alarm(0)
            break
        return res_t
    
    def _update_state_camera(self, cam_time=0):
        res_t = dict()
        signal.signal(signal.SIGALRM, self.__timeout_handler__)
        while True:
            try:
                try:
                    signal.alarm(1)
                    # signal.setitimer(signal.ITIMER_REAL, TIMEOUT)
                    renderer_key_set = []
                    for vehicle_key in self.vehicle_set.keys():
                        for camera_key in self.vehicle_set[vehicle_key]["model"].camera_info.keys():
                            pos_t = self.vehicle_set[vehicle_key]["model"].camera_pose[camera_key]["position"]
                            att_t = self.vehicle_set[vehicle_key]["model"].camera_pose[camera_key]["attitude"]
                            self.fg_renderer[self.camera_set[camera_key]["renderer"]].setCameraPose( \
                                pos_t, att_t, self.camera_set[camera_key]["index"])
                            self.camera_set[camera_key]["currentPos"] = pos_t
                            self.camera_set[camera_key]["currentAtt"] = att_t
                            if self.camera_set[camera_key]["renderer"] not in renderer_key_set:
                                renderer_key_set.append(self.camera_set[camera_key]["renderer"])
                    
                    for renderer_key in renderer_key_set:
                        self.fg_renderer[renderer_key].setStateTime(self.fg_renderer[renderer_key].getTimestamp())
                        self.fg_renderer[renderer_key].requestRender()
                        res, res_collision, res_landmark, res_lidar = self.fg_renderer[renderer_key].getImage()
                        for res_key in res.keys():
                            img = res[res_key]
                            img_reshape = np.reshape(img,(self.cam_height,self.cam_width,self.camera_set[res_key]["channels"]))
                            res_t[res_key] = img_reshape
                            # self.camera_set[res_key]["image"] = img_reshape
                            # img_reshape = cv2.flip(cv2.cvtColor(img_reshape, cv2.COLOR_BGR2RGB),-1)
                            # img_reshape = cv2.cvtColor(img_reshape, cv2.COLOR_BGR2RGB)
                            img_log = dict()
                            img_log["timestamp"] = cam_time
                            
                            file_path = os.path.join(self.camera_set[res_key]["img_dir"],"{0:019d}.png".format(np.uint64(cam_time*1e9)))
                            if (self.camera_set[res_key]["channels"] == 1):
                                cv2.imwrite(file_path, img_reshape.astype(np.uint16))
                            else:
                                cv2.imwrite(file_path, img_reshape.astype(np.uint8))

                            img_log["data"] = file_path
                            self.camera_set[res_key]["logs"].append(img_log)

                except Exception as e:
                    print(e)
                    continue
            except Exception as e:
                print(e)
                continue
            signal.alarm(0)
            break
        return res_t
    
    def plot_state(self, vehicle_id, attribute=None):
        logs = copy.deepcopy(self.vehicle_set[vehicle_id]["logs"])
        if self.vehicle_set[vehicle_id]["type"] == "uav":
            att_list = ["position", "velocity", "acceleration", "attitude_euler_angle", "angular_velocity", "angular_acceleration"]
            att_list2 = ["motor_speed", "motor_acceleration"]
            for att_ in att_list:
                if attribute == att_ or attribute == None:
                    time_array = []
                    data_array = []
                    for l in logs:
                        time_array.append(l["timestamp"])
                        data_array.append(l[att_])
                    time_array = np.array(time_array)
                    data_array = np.array(data_array)
                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    ax.plot(time_array, data_array[:,0], '-', label='x')
                    ax.plot(time_array, data_array[:,1], '-', label='y')
                    ax.plot(time_array, data_array[:,2], '-', label='z')
                    ax.set_title("{} - {}".format(vehicle_id, att_))
                    ax.set_xlabel('time (s)')
                    ax.legend()
                    ax.grid()

            for att_ in att_list2:
                if attribute == att_ or attribute == None:
                    time_array = []
                    data_array = []
                    for l in logs:
                        time_array.append(l["timestamp"])
                        data_array.append(l[att_])
                    time_array = np.array(time_array)
                    data_array = np.array(data_array)
                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    ax.plot(time_array, data_array[:,0], '-', label='motor_1')
                    ax.plot(time_array, data_array[:,1], '-', label='motor_2')
                    ax.plot(time_array, data_array[:,2], '-', label='motor_3')
                    ax.plot(time_array, data_array[:,3], '-', label='motor_4')
                    ax.set_title("{} - {}".format(vehicle_id, att_))
                    ax.set_xlabel('time (s)')
                    ax.legend()
                    ax.grid()
        
        elif self.vehicle_set[vehicle_id]["type"] == "car":
            att_list = ["position", "velocity"]
            att_list2 = ["heading", "speed"]
            for att_ in att_list:
                if attribute == att_ or attribute == None:
                    time_array = []
                    data_array = []
                    for l in logs:
                        time_array.append(l["timestamp"])
                        data_array.append(l[att_])
                    time_array = np.array(time_array)
                    data_array = np.array(data_array)
                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    ax.plot(time_array, data_array[:,0], '-', label='x')
                    ax.plot(time_array, data_array[:,1], '-', label='y')
                    ax.set_title("{} - {}".format(vehicle_id, att_))
                    ax.set_xlabel('time (s)')
                    ax.legend()
                    ax.grid()

            for att_ in att_list2:
                if attribute == att_ or attribute == None:
                    time_array = []
                    data_array = []
                    for l in logs:
                        time_array.append(l["timestamp"])
                        data_array.append(l[att_])
                    time_array = np.array(time_array)
                    data_array = np.array(data_array)
                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    ax.plot(time_array, data_array, '-')
                    ax.set_title("{} - {}".format(vehicle_id, att_))
                    ax.set_xlabel('time (s)')
                    ax.grid()
            # if flag_save:
            #     plt.savefig('{}/{}_der_{}.png'.format(save_dir,save_idx,der))
            #     plt.close()
        return
    
    def plot_state_camera(self, camera_id=None):
        for camera_key in self.camera_set.keys():
            if camera_key == camera_id or camera_id == None:
                if len(self.camera_set[camera_key]["logs"]) > 0:
                    plt.axis("off")
                    plt.tight_layout()
                    img_t = cv2.imread(self.camera_set[camera_key]["logs"][-1]["data"])
                    img_t = cv2.cvtColor(img_t, cv2.COLOR_BGR2RGB)
                    plt.imshow(img_t)
                    plt.title(camera_key)
                    plt.show()
                    plt.pause(0.1)
        return
    
    def plot_state_video(self, flag_save=False, filename="video", dpi=400):
        ani_set = dict()
        for camera_key in self.camera_set.keys():
            # if "image" in self.camera_set[camera_key].keys():
            if len(self.camera_set[camera_key]["logs"]) > 1:
                # print("plot key {} - len {}".format(camera_key, len(self.camera_set[camera_key]["logs"])))
                fig = plt.figure()
                plt.axis("off")
                plt.tight_layout()
                plt.margins(0)
                fig.subplots_adjust(bottom=0, top=1, left=0, right=1)
                # plt.imshow(cv2.cvtColor(self.camera_set[camera_key]["image"], cv2.COLOR_BGR2RGB))
                ims = []
                for img in self.camera_set[camera_key]["logs"]:
                    img_t = cv2.imread(img["data"])
                    img_t = cv2.cvtColor(img_t, cv2.COLOR_BGR2RGB)
                    im = plt.imshow(img_t, animated=True)
                    ims.append([im])
                ani = animation.ArtistAnimation( \
                    fig, ims, \
                    interval=1000./self.camera_set[camera_key]["freq"], blit=False, repeat_delay=1000)
                plt.close()
                ani_set[camera_key] = ani
                # HTML(ani.to_html5_video())
                if flag_save:
                    ani.save('{}_{}.mp4'.format(filename,camera_key), dpi=dpi)
        return ani_set
    
    def plot_state_map(self):
        if not os.path.exists(self.map_filepath):
            print("Cannot find map files from {}".format(self.map_filepath))
            return
        
        fig = go.Figure()
        # Add trace
        for vehicle_key in self.vehicle_set.keys():
            x_pos = [state["position"][1] for state in self.vehicle_set[vehicle_key]["logs"]]
            y_pos = [state["position"][0] for state in self.vehicle_set[vehicle_key]["logs"]]
            fig.add_trace(
                go.Scatter(
                    x=x_pos, y=y_pos,
                    name=vehicle_key,
                    marker=dict(
                        size=2,
                    ),
                )
            )

        img_width = np.int(self.map_y_max-self.map_y_min)
        img_height = np.int(self.map_x_max-self.map_x_min)
        img_margin = self.map_margin
        img_scale = self.map_scale
        # fig.add_trace(
        #     go.Scatter(
        #         x=[self.map_x_min, self.map_x_max],
        #         y=[self.map_y_min, self.map_y_max],
        #         mode="markers",
        #         marker_opacity=0
        #     )
        # )

        # Configure axes
        fig.update_xaxes(
            visible=True,
            range=[self.map_y_min, self.map_y_max]
        )

        fig.update_yaxes(
            visible=True,
            range=[self.map_x_min, self.map_x_max],
            scaleanchor="x"
        )
        
        map_img = Image.open(self.map_filepath)
        fig.add_layout_image(
            dict(
                source=map_img,
                xref="x",
                yref="y",
                x=self.map_y_min,
                y=self.map_x_max,
                sizex=img_width,
                sizey=img_height,
                sizing="stretch",
                opacity=1.0,
                layer="below")
        )

        # Set templates
        fig.update_layout(
            # template="plotly_white",
            width=img_width*img_scale+2*img_margin,
            height=img_height*img_scale+2*img_margin,
            margin={"l": img_margin, "r": img_margin, "t": img_margin, "b": img_margin},)

        fig.show(output_type='div', renderer="notebook")
        return

    def __timeout_handler__(self, signum, frame):
        raise Exception

if __name__ == "__main__":
    # execute only if run as a script
    env = flightgoggles_env()
    env.proceed_motor_speed("Vehicle_1", np.array([1100.0,1100.0,1100.0,1100.0]),0.1)
    env.plot_state()
    
