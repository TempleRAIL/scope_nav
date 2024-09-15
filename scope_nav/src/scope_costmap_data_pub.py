#!/usr/bin/env python
#
# revision history: xzt
#  20240604 (TE): first version
#
# usage: python scope_costmap_data_pub.py
#
# This script is the SCOPE Costmap code of the SCOPE-NAV navigation framework.
#------------------------------------------------------------------------------

from random import choice
import rospy
# custom define messages:
from scope_msgs.msg import ScopeInputData, ScopeOutputData
from geometry_msgs.msg import Point, PoseStamped, Twist, TwistStamped
from people_msgs.msg import People, Person
from nav_msgs.msg import Odometry, OccupancyGrid
from std_msgs.msg import Header
# python: 
import numpy as np
import math
# import the model and all of its variables/functions
#
from model import *
from local_occ_grid_map import LocalMap
import torch
import threading

# Constants
IMG_SIZE = 64 #80
SEQ_LEN = 10
NUM_CLASSES = 1
NUM_INPUT_CHANNELS = 1
NUM_LATENT_DIM = 512 #800 #512
NUM_OUTPUT_CHANNELS = NUM_CLASSES
NUM_TP = 10     # the number of timestamps

# Init map parameters
P_prior = 0.5	# Prior occupancy probability
P_occ = 0.7	    # Probability that cell is occupied with total confidence
P_free = 0.3	# Probability that cell is free with total confidence 
MAP_X_LIMIT = [0, 6.4]#[0, 8]      # Map limits on the x-axis
MAP_Y_LIMIT = [-3.2, 3.2]#[-4, 4]   # Map limits on the y-axis
RESOLUTION = 0.1        # Grid resolution in [m]'
TRESHOLD_P_OCC = 0.8    # Occupancy threshold

# for reproducibility, we seed the rng
#
set_seed(SEED1)        
# set the device to use GPU if available:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ScopeCostmap:
    # Constructor
    def __init__(self):
        # initialize data:  
        self.scan_ranges = np.zeros(1080)
        self.curr_vel = np.zeros(2)
        self.curr_pos = np.zeros(3)
        self.curr_odom = np.zeros(3)
        self.scope_header = Header()
        self.occ_grid = [] #np.zeros((IMG_SIZE, IMG_SIZE))
        # input:
        self.scans = []
        self.positions = []
        self.velocities = []
        self.header = Header() 

        # read truncnorm & skewcauchy model parameters:
        occ_entropy_path = rospy.get_param('~statistics_file', './model/truncnorm_skewcauchy_statistics_tables/truncnorm_skewcauchy_entropy_pred_time_6.npy')
        truncnorm_skewcauchy_occ_entropy = np.load(occ_entropy_path)
        self.c_entropy_table = torch.tensor(truncnorm_skewcauchy_occ_entropy).to(device)
        self.p_bins = torch.linspace(0, 1, steps=16).to(device)
        
        # initialize ROS objects
        self.scope_input_data_sub = rospy.Subscriber("scope_input_data", ScopeInputData, self.scope_input_data_callback)
    
        self.scope_output_data_pub = rospy.Publisher('scope_output_data', ScopeOutputData, queue_size=1, latch=False)
        self.scope_prediction_pub = rospy.Publisher('scope_prediction', People, queue_size=1, latch=False)
        self.scope_uncertainty_pub = rospy.Publisher('scope_uncertainty', People, queue_size=1, latch=False)
        self.local_map_pub = rospy.Publisher('local_map', OccupancyGrid, queue_size=1, latch=False)

        # So-SCOPE model:
        # instantiate a model:
        self.model = so_scope(input_channels=NUM_INPUT_CHANNELS,
                        latent_dim=NUM_LATENT_DIM,
                        output_channels=NUM_OUTPUT_CHANNELS)
        # moves the model to device (cpu in our case so no change):
        self.model.to(device)
        # set the model to evaluate
        #
        self.model.eval()
        # load the weights
        #
        model_file = rospy.get_param('~model_file', "./model/so_scope_model.pth")
        checkpoint = torch.load(model_file, map_location=device)
        self.model.load_state_dict(checkpoint['model'])
        print("Finish loading SO-SCOPE model.", device)

        # Lock
        self.lock = threading.Lock() # lock to keep twist/time thread safe

        # timer:
        self.ts_cnt = 0
        self.rate = 10  # 20 Hz velocity controller
        self.timer = rospy.Timer(rospy.Duration(1./self.rate), self.timer_callback)
    

    # Callback function for the local map subscriber
    def scope_input_data_callback(self, scope_input_data_msg):
        # get the local occupancy grid map data:
        self.scope_header = scope_input_data_msg.header
        self.scan_ranges = np.array(scope_input_data_msg.scan_ranges, dtype=np.float32)
        self.curr_vel = np.array(scope_input_data_msg.curr_vel, dtype=np.float32)
        self.curr_pos = np.array(scope_input_data_msg.curr_pos, dtype=np.float32)
        self.curr_odom = np.array(scope_input_data_msg.curr_odom, dtype=np.float32)

    # function that runs every time the timer finishes to ensure that vae data are sent regularly
    def timer_callback(self, event):  
        # collect 10 time step data:
        self.lock.acquire()
        self.header = self.scope_header
        self.scans.append(self.scan_ranges)
        self.positions.append(self.curr_pos)
        self.velocities.append(self.curr_vel)
        self.lock.release()

        self.ts_cnt = self.ts_cnt + 1

        if(self.ts_cnt == NUM_TP): 
            ## SOGMP inference:
            # collect the samples as a batch:
            scans = torch.FloatTensor(self.scans).unsqueeze(0)
            scans = scans.to(device)
            positions = torch.FloatTensor(self.positions).unsqueeze(0)
            positions = positions.to(device)
            velocities = torch.FloatTensor(self.velocities).unsqueeze(0)
            velocities = velocities.to(device)
            
            # create occupancy maps:
            batch_size = scans.size(0)
            # multi-step prediction: 10 time steps:
            # Create input grid maps: 
            input_gridMap = LocalMap(X_lim = MAP_X_LIMIT, 
                                    Y_lim = MAP_Y_LIMIT, 
                                    resolution = RESOLUTION, 
                                    p = P_prior,
                                    size=[batch_size, SEQ_LEN],
                                    device = device)
            # current position and velocities: 
            obs_pos_N = positions[:, SEQ_LEN-1]
            vel_N = velocities[:, SEQ_LEN-1]
            # Predict the future origin pose of the robot: t+n 
            T = 6 #SEQ_LEN #int(t_pred)
            noise_std = [0, 0, 0]#[0.00111, 0.00112, 0.02319]
            pos_origin = input_gridMap.origin_pose_prediction(vel_N, obs_pos_N, T, noise_std)
            # robot positions:
            pos = positions[:,:SEQ_LEN]
            # Transform the robot past poses to the predicted reference frame.
            x_odom, y_odom, theta_odom = input_gridMap.robot_coordinate_transform(pos, pos_origin)
            # Lidar measurements:
            distances = scans[:,:SEQ_LEN]
            # the angles of lidar scan: -135 ~ 135 degree
            angles = torch.linspace(-(135*np.pi/180), 135*np.pi/180, distances.shape[-1]).to(device)
            # Lidar measurements in X-Y plane: transform to the predicted robot reference frame
            distances_x, distances_y = input_gridMap.lidar_scan_xy(distances, angles, x_odom, y_odom, theta_odom)
            # discretize to binary maps:
            input_binary_maps = input_gridMap.discretize(distances_x, distances_y)
            
            # binary occupancy maps:
            input_binary_maps = input_binary_maps.unsqueeze(2)
            curr_map = input_binary_maps[:, -1].detach().cpu().numpy()
            
            # feed the batch to the network:
            num_samples = 1 
            inputs_samples = input_binary_maps.repeat(num_samples,1,1,1,1)

            for t in range(T):  
                prediction = self.model(inputs_samples)
                prediction = prediction.reshape(-1,1,1,IMG_SIZE,IMG_SIZE)
                inputs_samples = torch.cat([inputs_samples[:,1:], prediction], dim=1)

            predictions = prediction.detach().clone().squeeze(1)
            # mean and std:
            pred_mean = prediction.detach().clone().squeeze(1) 
            pred_entropy = torch.zeros((1, 1, IMG_SIZE, IMG_SIZE)).to(device)
            for k in range(15):
                c_entropy = self.c_entropy_table[k]
                idx = predictions <= self.p_bins[k+1]
                idx_size = torch.sum(idx==True)
                c_occ_entropys = -1*torch.ones(idx_size).to(device) * c_entropy
                pred_entropy[idx] = c_occ_entropys.to(torch.float32)
                predictions[idx] = 100
            
            ## pubish occupied people data: prediction
            occ_scope_pred = People()
            #occ_scope_pred.header = self.header
            occ_scope_pred.header.stamp = rospy.Time.now()
            occ_scope_pred.header.frame_id = "hokuyo_link"
            
            # get occupied indicies:
            pred_mean_occ = pred_mean.squeeze()
            pred_mean_occ[pred_mean_occ < 0.5] = 0
            idx_occ = torch.nonzero(pred_mean_occ)

            # translate grid indicies to the physical positions:
            px = MAP_X_LIMIT[0] + RESOLUTION*(idx_occ[:, 0] + 0.5)
            py = MAP_Y_LIMIT[0] + RESOLUTION*(idx_occ[:, 1] + 0.5)
            px = px.detach().cpu().numpy()
            py = py.detach().cpu().numpy()
            idx_occ = idx_occ.detach().cpu().numpy()

            for i in range(len(px)):
                o_pose = Person()
                o_pose.position.x = px[i]
                o_pose.position.y = py[i]
                o_pose.position.z = 0
                occ_scope_pred.people.append(o_pose)
            # publish prediction map:
            self.scope_prediction_pub.publish(occ_scope_pred)

            ## pubish occupied people data: uncertainty
            occ_scope_entropy = People()
            #occ_scope.header = self.header
            occ_scope_entropy.header.stamp = rospy.Time.now() 
            occ_scope_entropy.header.frame_id = "hokuyo_link"
            
            # get occupied indicies:
            pred_entropy_occ = pred_entropy.squeeze()
            pred_entropy_occ[pred_entropy_occ < 0.1] = 0
            idx_occ = torch.nonzero(pred_entropy_occ)

            # translate grid indicies to the physical positions:
            px = MAP_X_LIMIT[0] + RESOLUTION*(idx_occ[:, 0] + 0.5)
            py = MAP_Y_LIMIT[0] + RESOLUTION*(idx_occ[:, 1] + 0.5)
            px = px.detach().cpu().numpy()
            py = py.detach().cpu().numpy()
            idx_occ = idx_occ.detach().cpu().numpy()
     
            for i in range(len(px)):
                o_pose = Person()
                o_pose.position.x = px[i]
                o_pose.position.y = py[i]
                o_pose.position.z = 0
                occ_scope_entropy.people.append(o_pose)
            # publish uncertainty map:
            self.scope_uncertainty_pub.publish(occ_scope_entropy)

            ## get the output:
            pred_mean_map = pred_mean.detach().cpu().numpy()
            pred_entropy_map = pred_entropy.detach().cpu().numpy()

            # publish scope output data:
            prediction_map = np.concatenate((pred_mean_map, pred_entropy_map, curr_map), axis=1)
            self.occ_grid = prediction_map.reshape(-1).tolist()
            scope_output_data = ScopeOutputData()
            scope_output_data.occ_grid = [float(val) for val in self.occ_grid] #for subb in sublist for val in subb]
            self.scope_output_data_pub.publish(scope_output_data)
        
            # visualize the local occupancy map:
            # create message:
            occ_map = OccupancyGrid()
            # initialize header:
            #occ_map.header = self.header
            occ_map.header.stamp = rospy.Time.now()
            occ_map.header.frame_id = "hokuyo_link" #scan_msg.header.frame_id
            # initialize info:
            occ_map.info.map_load_time = rospy.Time.now()
            occ_map.info.resolution = RESOLUTION #xy_resolution
            occ_map.info.width = IMG_SIZE #width
            occ_map.info.height = IMG_SIZE #height
            occ_map.info.origin.position.x = MAP_X_LIMIT[0] #min_x
            occ_map.info.origin.position.y = MAP_Y_LIMIT[0]# min_y
            # initialize data:
            occ_pred_map = pred_mean_map.squeeze().transpose().reshape(-1).tolist()
            occ_map.data = [int(val*100) for val in occ_pred_map]#.transpose() for val in sublist]
            # print map information to screen
            #rospy.loginfo("local map info: resolution: %.2f, width: %.2f, height: %.2f", 
            #            occ_map.info.resolution, occ_map.info.width, occ_map.info.height)
            # publish local map msg:
            self.local_map_pub.publish(occ_map)

            # reset the position data list:
            self.ts_cnt = NUM_TP-1
            self.scans = self.scans[1:NUM_TP]
            self.positions = self.positions[1:NUM_TP]
            self.velocities = self.velocities[1:NUM_TP]
        
        
if __name__ == '__main__':
    rospy.init_node('scope_nav')
    scope_costmap = ScopeCostmap()
    rospy.spin()



