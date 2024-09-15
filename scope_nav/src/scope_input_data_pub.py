#!/usr/bin/env python
#
# revision history: xzt
#  20240604 (TE): first version
#
# usage: python scope_input_data_pub.py
#
# This script is the SCOPE input code of the SCOPE-NAV navigation framework.
#------------------------------------------------------------------------------

import message_filters
from random import choice
import rospy
import tf
# custom define messages:
from scope_msgs.msg import ScopeInputData
from geometry_msgs.msg import Point, PoseStamped, Twist, TwistStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, OccupancyGrid
from std_msgs.msg import Header
# python: 
import numpy as np
import math
import threading


class ScopeInputDataPub:
    # Constructor
    def __init__(self):
        # initialize data:  
        self.scan_ranges = np.zeros(1080)
        self.curr_vel = np.zeros(2)
        self.curr_pos = np.zeros(3)
        self.curr_odom = np.zeros(3)
        self.header = Header()

        # initialize ROS objects
        self.tf_listener = tf.TransformListener()
        self.scan_sub = message_filters.Subscriber("scan", LaserScan)
        self.robot_vel_sub = message_filters.Subscriber("odom", Odometry)#('/mobile_base/commands/velocity', Twist) #, queue_size=1)
        self.scope_data = message_filters.ApproximateTimeSynchronizer([self.scan_sub, self.robot_vel_sub], queue_size=5, slop=0.5, allow_headerless=True)
        self.scope_data.registerCallback(self.scope_data_callback)

        self.scope_input_data_pub = rospy.Publisher('scope_input_data', ScopeInputData, queue_size=1, latch=False)

        # Lock
        self.lock = threading.Lock() # lock to keep twist/time thread safe

        # timer:
        self.ts_cnt = 0
        self.rate = 20  # 20 Hz velocity controller
        self.timer = rospy.Timer(rospy.Duration(1./self.rate), self.timer_callback)

    # Get the current pose of the robot from the tf tree
    def get_current_pose(self):
        trans = rot = None
        trans_odom = rot_odom = None
        # look up the current pose of the base_footprint using the tf tree
        try:
            (trans,rot) = self.tf_listener.lookupTransform('/map', '/base_link', rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logwarn('Could not get robot pose')
            return self.curr_pos
        '''
        try:
            (trans_odom,rot_odom) = self.tf_listener.lookupTransform('/odom', '/base_link', rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logwarn('Could not get robot odom')
            return self.curr_odom
        '''
        # pose:
        (roll, pitch, theta) = tf.transformations.euler_from_quaternion(rot)
        rospy.logdebug("x = {}, y = {}, theta = {}".format(trans[0], trans[1], theta))
        #robot_pose = np.array([trans[0], trans[1], theta])

        # odom:
        #(roll_odom, pitch_odom, theta_odom) = tf.transformations.euler_from_quaternion(rot_odom)
        #rospy.logdebug("x = {}, y = {}, theta = {}".format(trans_odom[0], trans_odom[1], theta_odom))
        #robot_odom= np.array([trans_odom[0], trans_odom[1], theta_odom])

        robot_pose_odom = np.array([trans[0], trans[1], theta, 0, 0, 0])#trans_odom[0], trans_odom[1], theta_odom])

        return robot_pose_odom

    # Callback function for the local map subscriber
    def scope_data_callback(self, scan_msg, robot_vel_msg):
        # get the local occupancy grid map data:
        self.header = scan_msg.header
        self.scan_ranges = np.array(scan_msg.ranges, dtype=np.float32)
        self.scan_ranges[np.isnan(self.scan_ranges)] = 20.
        self.scan_ranges[np.isinf(self.scan_ranges)] = 20. 

        self.curr_vel[0] = robot_vel_msg.twist.twist.linear.x
        self.curr_vel[1] = robot_vel_msg.twist.twist.angular.z
        '''
        robot_pose_odom = self.get_current_pose()
        self.curr_pos = robot_pose_odom[:3]
        #self.curr_odom = robot_pose_odom[3:]
        # publish vae input data:
        scope_input_data_msg = ScopeInputData()
        scope_input_data_msg.scan_ranges = self.scan_ranges 
        scope_input_data_msg.curr_vel = self.curr_vel
        scope_input_data_msg.curr_pos = self.curr_pos
        scope_input_data_msg.curr_odom = self.curr_odom
        self.scope_input_data_pub.publish(scope_input_data_msg)
        '''

    # function that runs every time the timer finishes to ensure that vae data are sent regularly
    def timer_callback(self, event):  
        robot_pose_odom = self.get_current_pose()
        self.curr_pos = robot_pose_odom[:3]
        #self.curr_odom = robot_pose_odom[3:]
        # publish vae input data:
        scope_input_data_msg = ScopeInputData()
        self.lock.acquire()
        scope_input_data_msg.header = self.header
        scope_input_data_msg.scan_ranges = self.scan_ranges 
        scope_input_data_msg.curr_vel = self.curr_vel
        self.lock.release()
        scope_input_data_msg.curr_pos = self.curr_pos
        scope_input_data_msg.curr_odom = self.curr_odom
        self.scope_input_data_pub.publish(scope_input_data_msg)


if __name__ == '__main__':
    rospy.init_node('scope_input_data_pub')
    scope_input_data = ScopeInputDataPub()
    rospy.spin()



