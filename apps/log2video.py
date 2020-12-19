#!/usr/bin/env python
# coding: utf-8
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Slerp.html
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html#scipy.spatial.transform.Rotation

import numpy as np
import pandas as pd
import cv2
import signal
import os, sys, time, copy, argparse
from progress.bar import Bar

from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from scipy import interpolate

curr_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(curr_path,'../'))
from flightgoggles.utils import *
from flightgoggles.env import flightgoggles_env

parser = argparse.ArgumentParser(description='FlightGoggles log2video')
parser.add_argument('-f', "--file_path", help="assign log file path", default=os.path.join(curr_path,"example_log.csv"))
parser.add_argument('-o', "--output", help="assign output video name", default=os.path.join(curr_path,"test.avi"))
args = parser.parse_args()
print("Reading log file from {}".format(args.file_path))

data = pd.read_csv(args.file_path, sep=',', header=None).values[1:,:]
data = np.double(data)
save_path = os.path.join(curr_path,"./tmp")
if not os.path.exists(save_path):
    os.makedirs(save_path)

env = flightgoggles_env(cfg_fgclient="FlightGogglesClient_debug_env.yaml")
# pos_curr = env.get_state("uav1")["position"]
# yaw_curr = env.get_state("uav1")["attitude_euler_angle"][2]

FPS_VIDEO = 60

# Interpolation
time_array = data[:,0]*1e-9
total_time = time_array[-1]
t_new = np.arange(time_array[0], total_time, 1.0/FPS_VIDEO)

fx = interpolate.interp1d(time_array, data[:,1], fill_value="extrapolate")
fy = interpolate.interp1d(time_array, data[:,2], fill_value="extrapolate")
fz = interpolate.interp1d(time_array, data[:,3], fill_value="extrapolate")
pos_new = np.zeros((t_new.shape[0], 3))
pos_new[:,0] = fx(t_new)
pos_new[:,1] = fy(t_new)
pos_new[:,2] = fz(t_new)

att_array = np.empty((0,4))
for i in range(data.shape[0]):
    att_array = np.append(att_array, quat_wx2xw(Euler2quat(data[i,7:10]))[np.newaxis,:], axis=0)
key_rots = R.from_quat(att_array)
slerp = Slerp(time_array, key_rots)
interp_rots = slerp(t_new)
att_new_array = R.as_quat(interp_rots)
att_new = np.zeros_like(att_new_array)
for i in range(t_new.shape[0]):
    att_new[i,:] = quat_xw2wx(att_new_array[i,:])

# Progress Bar
data_len = t_new.shape[0]
print("data length: {}".format(data_len))
bar_iter = 0
bar_max = data_len
bar = Bar('Processing Video', max=bar_max, suffix='%(percent)d%%')
bar_step = np.around(data_len/bar_max)

# Request Image and Save
for i in range(data_len):
    filename = "{}/{}.png".format(save_path,i)
    if (np.around(data_len/bar_step) > bar_iter):
        bar.next()
    env.set_state_camera("cam0", pos_new[i,:], att_new[i,:],flag_save_logs=True)
    img = env.get_camera_image("cam0")[-1]["data"]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # pos_t, att_t = ned2eun(pos=pos_new[i,:],att=att_new[i,:])
    # img = fgc.get_image(pos=pos_t,att=att_t)
    cv2.imwrite(filename,img)
height, width, layers = img.shape
size = (width,height)
bar.finish()
env.close()

# Load image any generate video
img_array = []
for i in range(data_len):
    filename = os.path.join(save_path,"{}.png".format(i))
    print(filename)
    img_t = cv2.imread(filename)
    height, width, layers = img_t.shape
    size = (width,height)
    img_array.append(img_t)

out = cv2.VideoWriter(
    filename=args.output, 
    fourcc=cv2.VideoWriter_fourcc(*'DIVX'), 
    fps=FPS_VIDEO, 
    frameSize=size)

for i in range(data_len):
    out.write(img_array[i])
out.release()

