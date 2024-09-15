#!/usr/bin/env python
#
# file: $ISIP_EXP/SOGMP/scripts/local_occ_grid_map.py
#
# revision history: xzt
#  20220824 (TE): first version
#
# usage:
#
# arguments:
#  X_lim: map limits on the x-axis
#  Y_lim: map limits on the y-axis
#  resolution: grid resolution in [m]'
#  p: Prior occupancy probability
#  size: the size of input lidar measurements: batch * time
#  device: the device to use (GPU or CPU)
#
# This script is a GPU-accelerated and parallelized occupancy grid mapping algorithm that parallelizes the independent cell state update operations, written in pytorch.
#------------------------------------------------------------------------------

# import pytorch modules
#
import torch
from bresenham_torch import bresenhamline

#
# end of function


class LocalMap:
    # Constructor
    def __init__(self, X_lim, Y_lim, resolution, p, size=[1, 1], device=None):
        # initialize parameters:  
        self.X_lim = X_lim
        self.Y_lim = Y_lim
        self.resolution = resolution
        self.size = size
        self.device = device

        x = torch.arange(start = X_lim[0], end = X_lim[1], step = resolution)
        y = torch.arange(start = Y_lim[0], end = Y_lim[1], step = resolution)

        self.x_max = len(x)
        self.y_max = len(y)
        
        # probability matrix in log-odds scale:
        self.occ_map = torch.full((self.size[0], self.size[1], self.x_max, self.y_max), fill_value = self.log_odds(p))
        if(self.device is not None):
            self.occ_map = self.occ_map.to(self.device)

    def log_odds(self, p):
        """
        Log odds ratio of p(x):

                    p(x)
        l(x) = log ----------
                    1 - p(x)

        """
        p = torch.tensor(p)

        return torch.log(p / (1 - p))


    def retrieve_p(self, log_map):
        """
        Retrieve p(x) from log odds ratio:

                        1
        p(x) = 1 - ---------------
                    1 + exp(l(x))

        """
        prob_map = 1 - 1 / (1 + torch.exp(log_map))

        return prob_map

    def lidar_scan_xy(self, distances, angles, x_odom, y_odom, theta_odom):
        """
        Lidar measurements in X-Y plane
        """
        # expand the tensor to the same size of scans:
        angles = angles.expand(distances.size(0), distances.size(1), distances.size(2))
        x_odom = x_odom.unsqueeze(2).expand(distances.size(0), distances.size(1), distances.size(2))
        y_odom = y_odom.unsqueeze(2).expand(distances.size(0), distances.size(1), distances.size(2))
        theta_odom = theta_odom.unsqueeze(2).expand(distances.size(0), distances.size(1), distances.size(2))

        # lidar scans to cartesian obstacles:
        distances_x = x_odom + distances * torch.cos(angles + theta_odom)
        distances_y = y_odom + distances * torch.sin(angles + theta_odom)

        return distances_x, distances_y
    
    def is_valid(self, x_r, y_c):
        """
        Flag of valid grid indicies
        """
        flag_v = (x_r < self.x_max) & (y_c < self.y_max) & (x_r >= 0) & (y_c >= 0)
            
        return flag_v

    def discretize(self, x, y):
        """
        Discretize continious x and y 
        """
        # translate the physical position to grid indicies
        x_r = torch.floor((x - self.X_lim[0]) / self.resolution).to(int)
        y_c = torch.floor((y - self.Y_lim[0]) / self.resolution).to(int)
        # get valid batch indicies:
        flag_v = self.is_valid(x_r, y_c)
        idx_v = torch.nonzero(flag_v)
        # get valid grid indicies:
        x_rv = x_r[idx_v[:,0],idx_v[:,1],idx_v[:,2]]
        y_cv = y_c[idx_v[:,0],idx_v[:,1],idx_v[:,2]]
        # create a binary map:
        binary_map = torch.zeros(self.size[0], self.size[1], self.x_max, self.y_max)
        if(self.device is not None):
            binary_map = binary_map.to(self.device)

        binary_map[idx_v[:,0], idx_v[:,1], x_rv, y_cv] = 1

        return binary_map

    def update(self, x0, y0, x, y, p_free, p_occ):
        """
        Update x and y coordinates in discretized grid map
        """
        # discretize: 
        binary_map = self.discretize(x, y)

        # find free space:
        occ_map = binary_map.clone().detach()
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                end = torch.nonzero(binary_map[i, j])
                if(end.size(0) != 0): # has obstacles
                    # start point:
                    x0_r = torch.floor((x0[i,j] - self.X_lim[0]) / self.resolution).to(int)
                    y0_c = torch.floor((y0[i,j] - self.Y_lim[0]) / self.resolution).to(int)
                    start = torch.tensor([x0_r, y0_c]).to(self.device)
                    start = start.unsqueeze(1).expand(end.size(1), end.size(0)).permute(1, 0)
                    # get free points from bresenham algorithm: 
                    points = bresenhamline(end, start, max_iter=-1)
                    x_r = points[:, 0]
                    y_c = points[:, 1]
                    flag_v = self.is_valid(x_r, y_c)
                    idx_v = torch.nonzero(flag_v)
                    # get valid grid indicies:
                    x_rv = x_r[idx_v[:,0]]
                    y_cv = y_c[idx_v[:,0]]
                    occ_map[i, j, x_rv, y_cv] = -1
    
        # update probability matrix using inverse sensor model
        self.occ_map[occ_map==-1] += self.log_odds(p_free)
        self.occ_map[occ_map==1] += self.log_odds(p_occ)

    
    def calc_MLE(self, prob_map, threshold_p_occ):
        """
        Calculate Maximum Likelihood estimate of the map (binary map)
        """
        prob_map[prob_map >= threshold_p_occ] = 1
        prob_map[prob_map < threshold_p_occ] = 0

        return prob_map
    
    def to_prob_occ_map(self, threshold_p_occ):
        """
        Transformation to GRAYSCALE image format
        """
        log_map = torch.sum(self.occ_map, dim=1)  # sum of all timestep maps
        prob_map = self.retrieve_p(log_map)
        prob_map = self.calc_MLE(prob_map, threshold_p_occ)

        return prob_map
    
    def origin_pose_prediction(self, vel_N, obs_pos_N, T, noise_std=[0,0,0]):
        """
        Predict the future origin pose of the robot: find the predicted reference frame
        """
        pos_origin = torch.zeros(self.size[0], 3)
        # Gaussian noise sampled from a distribution with MEAN=0.0 and STD=std
        x_noise = torch.randn(self.size[0])*noise_std[0]
        y_noise = torch.randn(self.size[0])*noise_std[1]
        th_noise = torch.randn(self.size[0])*noise_std[2]
        if(self.device is not None):
            pos_origin = pos_origin.to(self.device)
            x_noise = x_noise.to(self.device)
            y_noise = y_noise.to(self.device)
            th_noise = th_noise.to(self.device)
        # r_Ao_No: predicted reference position: T = t+n th timestep
        d = vel_N[:, 0]*0.1*T
        theta = vel_N[:, 1]*0.1*T
        pos_origin[:, 0] = obs_pos_N[:, 0] + d*torch.cos(obs_pos_N[:, 2]) + x_noise
        pos_origin[:, 1] = obs_pos_N[:, 1] + d*torch.sin(obs_pos_N[:, 2]) + y_noise
        pos_origin[:, 2] = obs_pos_N[:, 2] + theta + th_noise

        return pos_origin

    def robot_coordinate_transform(self, pos, pos_origin):
        """
        Transform the robot past poses to the predicted reference frame.
        """
        # expand the tensor to the same size of pos:
        pos_origin = pos_origin.unsqueeze(2).expand(pos.size(0), pos.size(2), pos.size(1)).permute(0,2,1)
        # Odometry measurements
        dx = pos[:, :, 0] - pos_origin[:, :, 0]
        dy = pos[:, :, 1] - pos_origin[:, :, 1]
        th = pos_origin[:, :, 2]
        x_odom = torch.cos(th) * dx + torch.sin(th) * dy
        y_odom = torch.sin(-th) * dx + torch.cos(th) * dy
        theta_odom = pos[:, :, 2] - th

        return x_odom, y_odom, theta_odom
