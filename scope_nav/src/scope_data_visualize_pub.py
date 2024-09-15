#!/usr/bin/env python
#
# revision history: xzt
#  20240604 (TE): first version
#
# usage: python scope_data_visualize_pub.py
#
# This script is the SCOPE data visulization code of the SCOPE-NAV navigation framework.
#------------------------------------------------------------------------------

import rospy
import geometry_msgs.msg
from geometry_msgs.msg import TwistStamped, Twist, PoseStamped, Pose
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from scope_msgs.msg import ScopeInputData, ScopeOutputData
from cv_bridge import CvBridge

import numpy as np
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages') # in order to import cv2 under python3
import cv2
import matplotlib.pyplot as plt


# get the coloar map:
def get_mpl_colormap(cmap_name):
    cmap = plt.get_cmap(cmap_name)
    # Initialize the matplotlib color map
    sm = plt.cm.ScalarMappable(cmap=cmap)
    # Obtain linear color range
    color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:,2::-1]
    return color_range.reshape(256, 1, 3)

class ScopeImageVisualizer:
    # ped data:
    #track_ped = None # pedestrian's information

    # ROS objects
    ped_sub = None # subscriber to get the global path
    tf_listener = None # tf listener to get the pose of the robot
    track_ped_pub = None # publisher to send the velocity commands
    
    # Constructor
    def __init__(self): 
        self.bridge = CvBridge()

        # Initialize ROS objects
        self.scope_sub = rospy.Subscriber('scope_output_data', ScopeOutputData, self.scope_callback)
        self.prediction_pub = rospy.Publisher('prediction_img', Image, queue_size=10)
        self.uncertainty_pub = rospy.Publisher('uncertainty_img', Image, queue_size=10)

    # Callback function for the path subscriber
    def scope_callback(self, vae_msg):
        scope_data = vae_msg.occ_grid
        # prediction:
        scope_prediction = np.array(scope_data[:64*64])
        scope_prediction = scope_prediction.reshape(64, 64)
        scope_prediction = np.flip(scope_prediction)#.transpose()) #.reshape(80,80,1)
        prediction_img = cv2.applyColorMap(np.uint8(scope_prediction*255), cv2.COLORMAP_BONE)#, get_mpl_colormap('PuRd')) # 'PuRd', 'binary', 'gist_heat_r'
        # uncertainty:
        scope_uncertainty = np.array(scope_data[64*64*2:])
        scope_uncertainty = scope_uncertainty.reshape(64, 64)
        scope_uncertainty = np.flip(scope_uncertainty)#.transpose()) #.reshape(80,80,1)
        uncertainty_img = cv2.applyColorMap(np.uint8(scope_uncertainty*255), cv2.COLORMAP_BONE)#, get_mpl_colormap('PuRd')) # 'PuRd', 'binary', 'gist_heat_r'

        # publish the data:
        self.prediction_pub.publish(self.bridge.cv2_to_imgmsg(prediction_img, encoding="passthrough"))
        self.uncertainty_pub.publish(self.bridge.cv2_to_imgmsg(uncertainty_img, encoding="passthrough"))
        
 
if __name__ == '__main__':
    rospy.init_node('scope_visualizer')

    scope_visualizer = ScopeImageVisualizer()
        
    rospy.spin()
