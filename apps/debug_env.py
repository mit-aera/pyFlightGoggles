#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
import cv2
import signal
import os, sys, time, copy, argparse
from scipy.spatial.transform import Rotation as R

curr_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(curr_path,'../'))
from flightgoggles.utils import *
from flightgoggles.env import flightgoggles_env

def _find_getch():
    try:
        import termios
    except ImportError:
        # Non-POSIX. Return msvcrt's (Windows') getch.
        import msvcrt
        return msvcrt.getch

    import sys, tty
    def _getch():
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

    return _getch

getch = _find_getch()

def timeout_handler(signum, frame):
    raise Exception

if __name__ == "__main__":
    # execute only if run as a script
    env = flightgoggles_env(cfg_fgclient="FlightGogglesClient_debug_env.yaml")
    
    pos_curr = env.get_state("uav1")["position"]
    yaw_curr = env.get_state("uav1")["attitude_euler_angle"][2]
    env.set_state_camera("cam0", \
        pos_curr, \
        Euler2quat(np.array([0.,0.,yaw_curr])))
    print("position: [{}]".format(', '.join([str(x) for x in pos_curr])))
    print("yaw: {}".format(yaw_curr))
    curr_unit = 0.1
    curr_unit_yaw = 5*np.pi/180

    while True:
        print("#########################################")
        print("current unit: {}, current unit yaw: {}".format(curr_unit, curr_unit_yaw))
        print("action: ")
        
        signal.signal(signal.SIGALRM, timeout_handler)
        while True:
            signal.alarm(1)
            try:
                key = getch()
                print(key)
                if key == "q":
                    print("exit")
                    sys.exit(0)
                elif key == "w":
                    pos_curr[2] -= curr_unit
                elif key == "s":
                    pos_curr[2] += curr_unit
                elif key == "a":
                    yaw_curr -= curr_unit_yaw
                elif key == "d":
                    yaw_curr += curr_unit_yaw
                elif key == "i":
                    pos_curr[0] += curr_unit*np.cos(yaw_curr)
                    pos_curr[1] += curr_unit*np.sin(yaw_curr)
                elif key == "k":
                    pos_curr[0] -= curr_unit*np.cos(yaw_curr)
                    pos_curr[1] -= curr_unit*np.sin(yaw_curr)
                elif key == "j":
                    pos_curr[0] -= -curr_unit*np.sin(yaw_curr)
                    pos_curr[1] -= curr_unit*np.cos(yaw_curr)
                elif key == "l":
                    pos_curr[0] += -curr_unit*np.sin(yaw_curr)
                    pos_curr[1] += curr_unit*np.cos(yaw_curr)
                elif key == "1":
                    curr_unit += 0.1
                elif key == "2":
                    curr_unit -= 0.1
                elif key == "3":
                    curr_unit_yaw += 5
                elif key == "4":
                    curr_unit_yaw -= 5

                yaw_curr %= 2*np.pi
                # r = R.from_euler('z', yaw_curr, degrees=True)
                # att = quat_xw2wx(R.as_quat(r))
                
                print("position: [{}]".format(', '.join([str(x) for x in pos_curr])))
                print("yaw: {}".format(yaw_curr))

                curr_pos_ = np.array([pos_curr[0],pos_curr[1],pos_curr[2],yaw_curr])
                signal.alarm(1)
                env.set_state_camera("cam0", \
                    pos_curr, \
                    Euler2quat(np.array([0.,0.,yaw_curr])), flag_update_simulation=False)
            except Exception:
                curr_pos_ = np.array([pos_curr[0],pos_curr[1],pos_curr[2],yaw_curr])
                signal.alarm(1)
                env.set_state_camera("cam0", \
                    pos_curr, \
                    Euler2quat(np.array([0.,0.,yaw_curr])), flag_update_simulation=False)
                continue
            
            break
