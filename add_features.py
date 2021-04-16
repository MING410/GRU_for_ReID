import numpy as np
import math
import copy
from sklearn.externals import joblib
import pyrealsense2 as rs

class add_features:

    def __init__(self):
        self.head_h = 0
        self.head_variability = 0 
        self.upper= 0
        self.left_angle = 0
        self.right_angle = 0
        self.gc_kneelength_rt = 0
        self.gc_kneelength = 0
        self.gc_elbowlength_rt = 0
        self.gc_elbowlength = 0
        self.gc_wristlength_rt = 0
        self.gc_wristlength = 0

    def head(self):
       #lastn_kp_xyzはアルファポーズからのデータ
        lastn_length = len(self.lastn_kp_xyz)
        current_kp_xyz = self.lastn_kp_xyz[lastn_length-1]
        past_kp_xyz = self.lastn_kp_xyz[lastn_length-2]
        #print('head calculation: ',current_kp_xyz)
        #print('head calculation: ', past_kp_xyz) 
        lefteye_current_location = current_kp_xyz[1]
        lefteye_last_location = past_kp_xyz[1]
        righteye_current_location = current_kp_xyz[2]
        righteye_last_location = past_kp_xyz[2]
        if (lefteye_current_location[0]**2+lefteye_current_location[1]**2) <= (righteye_current_location[0]**2+righteye_current_location[1]**2):
            if (math.isnan(lefteye_current_location[2]) == False) and (
                        math.isnan(lefteye_last_location[2]) == False):
                self.head_h = lefteye_current_location[2] - lefteye_last_location[2]
        else:
            if (math.isnan(righteye_current_location[2]) == False) and (
                        math.isnan(righteye_last_location[2]) == False):
                self.head_h = righteye_current_location[2] - righteye_last_location[2]
        return self.head_h
    #upper
    def upperpose(self):

       #lastn_kp_xyzはアルファポーズからのデータ
        lastn_length = len(self.lastn_kp_xyz)
        current_kp_xyz = self.lastn_kp_xyz[lastn_length-1]
        left_shoulder_xyz = current_kp_xyz[5]
        right_shoulder_xyz = current_kp_xyz[6]
        left_hip_xyz = current_kp_xyz[11]
        right_hip_xyz = current_kp_xyz[12]
        if (left_shoulder_xyz[0]**2+left_shoulder_xyz[1]**2) <= (right_shoulder_xyz[0]**2+right_shoulder_xyz[1]**2):
            if (math.isnan(left_shoulder_xyz[0]) == False) and (
                    math.isnan(left_hip_xyz[0]) == False):
                 self.upper = ((left_shoulder_xyz[0]-left_hip_xyz[0])**2+(left_shoulder_xyz[1]-left_hip_xyz[1])**2+(left_shoulder_xyz[2]-left_hip_xyz[2])**2)**0.5
                 self.upper = round(self.upper,3)
            # if  > 900:
            #     steplength_rt = 0
        else:
            if (math.isnan(right_shoulder_xyz[0]) == False) and (
                    math.isnan(right_hip_xyz[0]) == False):
                self.upper = ((right_shoulder_xyz[0]-right_hip_xyz[0])**2+(right_shoulder_xyz[1]-right_hip_xyz[1])**2+(right_shoulder_xyz[2]-right_hip_xyz[2])**2)**0.5
                self.upper = round(self.upper,3)
        return self.upper

    def angle(self):
        lastn_length = len(self.lastn_kp_xyz)
        current_kp_xyz = self.lastn_kp_xyz[lastn_length-1]
        left_knee_xyz = current_kp_xyz[13]
        right_knee_xyz = current_kp_xyz[15]
        left_ankle_xyz = current_kp_xyz[14]
        right_ankle_xyz = current_kp_xyz[16]
        if (math.isnan(left_knee_xyz[0]) == False) and (math.isnan(left_ankle_xyz[0]) == False):
            self.left_angle = math.atan2(((left_knee_xyz[0]-left_ankle_xyz[0])**2+(left_knee_xyz[1]-left_ankle_xyz[1])**2)**0.5,left_knee_xyz[2]-left_ankle_xyz[2])
            self.left_angle =round(self.left_angle,3)
        if (math.isnan(right_knee_xyz[0]) == False) and (math.isnan(right_ankle_xyz[0]) == False):
            self.right_angle = math.atan2(((right_knee_xyz[0]-right_ankle_xyz[0])**2+(right_knee_xyz[1]-right_ankle_xyz[1])**2)**0.5,right_knee_xyz[2]-right_ankle_xyz[2])
            self.right_angle =round(self.right_angle,3)
        return self.left_angle,self.right_angle

    def knee_calculation(self):
       
        kneelength_rt = 0
        lastn_length = len(self.lastn_kp_xyz)
        left_knee_xyz = self.lastn_kp_xyz[lastn_length-1][13]
        right_knee_xyz = self.lastn_kp_xyz[lastn_length - 1][14]

        #print('left_knee_xyz: ',left_knee_xyz)
        #print('right_knee_xyz: ',right_knee_xyz)
        
        if (math.isnan(left_knee_xyz[0]) == False) and (
                    math.isnan(right_knee_xyz[0]) == False):
            kneelength_rt = ((left_knee_xyz[0]-right_knee_xyz[0])**2+(left_knee_xyz[1]-right_knee_xyz[1])**2+(left_knee_xyz[2]-right_knee_xyz[2])**2)**0.5
            if kneelength_rt > 900:
                kneelength_rt = 0
        self.lastn_gc_kneelength.append(kneelength_rt/1000)
        #print('lastn_gc_kneelength: ',self.lastn_gc_kneelength)
        #print('lastn_gc_steplength: ',self.lastn_gc_steplength)
        if len(self.lastn_gc_kneelength) > self.step_time_interval:
            del self.lastn_gc_kneelength[0]
        #self.gc_steplength = round(steplength_rt/1000,3) #current step length
        self.gc_kneelength = round(max(self.lastn_gc_kneelength),3)
        self.gc_kneelength_rt = round(self.lastn_gc_kneelength[-1],3)
        return self.gc_kneelength,self.gc_kneelength_rt

    def elbow_calculation(self):
        
        elbowlength_rt = 0
        lastn_length = len(self.lastn_kp_xyz) 
        ##lastn_gc_leftelbow_distance##
        left_elbow_xyz = self.lastn_kp_xyz[lastn_length-1][7]
        right_elbow_xyz = self.lastn_kp_xyz[lastn_length - 1][8]

        #print('left_elbow_xyz: ',left_elbow_xyz)
        #print('right_elbow_xyz: ',right_elbow_xyz)
        
        if (math.isnan(left_elbow_xyz[0]) == False) and (
                    math.isnan(right_elbow_xyz[0]) == False):
            elbowlength_rt = ((left_elbow_xyz[0]-right_elbow_xyz[0])**2+(left_elbow_xyz[1]-right_elbow_xyz[1])**2+(left_elbow_xyz[2]-right_elbow_xyz[2])**2)**0.5
            if elbowlength_rt > 900:
                elbowlength_rt = 0
        self.lastn_gc_elbowlength.append(elbowlength_rt/1000)
        #print('lastn_gc_elbowlength: ',self.lastn_gc_kneelength)
        #print('lastn_gc_steplength: ',self.lastn_gc_steplength)
        if len(self.lastn_gc_elbowlength) > self.step_time_interval:
            del self.lastn_gc_elbowlength[0]
        #self.gc_steplength = round(steplength_rt/1000,3) #current step length
        self.gc_elbowlength = round(max(self.lastn_gc_elbowlength),3)
        self.gc_elbowlength_rt = round(self.lastn_gc_elbowlength[-1],3)
        return self.gc_elbowlength,self.gc_elbowlength_rt

    def wrist_calculation(self):
        
        wristlength_rt = 0
        lastn_length = len(self.lastn_kp_xyz)
        left_wrist_xyz = self.lastn_kp_xyz[lastn_length-1][9]
        right_wrist_xyz = self.lastn_kp_xyz[lastn_length - 1][10]

        #print('left_wrist_xyz: ',left_wrist_xyz)
        #print('right_wrist_xyz: ',right_wrist_xyz)
        
        if (math.isnan(left_wrist_xyz[0]) == False) and (
                    math.isnan(right_wrist_xyz[0]) == False):
            wristlength_rt = ((left_wrist_xyz[0]-right_wrist_xyz[0])**2+(left_wrist_xyz[1]-right_wrist_xyz[1])**2+(left_wrist_xyz[2]-right_wrist_xyz[2])**2)**0.5
            if wristlength_rt > 900:
                wristlength_rt = 0
       #len(lastn_gc_elbowlength )?
        self.lastn_gc_wristlength.append(wristlength_rt/1000)
        #print('lastn_gc_wristlength: ',self.lastn_gc_wristlength)
        #print('lastn_gc_wristlength: ',self.lastn_gc_wristlength)
        if len(self.lastn_gc_wristlength) > self.step_time_interval:
            del self.lastn_gc_wristlength[0]
        #self.gc_steplength = round(steplength_rt/1000,3) #current step length
        self.gc_wristlength = round(max(self.lastn_gc_wristlength),3)
        self.gc_wristlength_rt = round(self.lastn_gc_wristlength[-1],3)
        return self.gc_wristlength,self.gc_wristlength_rt



