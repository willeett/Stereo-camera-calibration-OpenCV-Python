# Stereo-camera-calibration-OpenCV-Python

camera_calibrate.py (Latest update)
Stereo camera calibration and creation of disparity map. Takes left and right images of a chessboard and calibrates the cameras and computes the parameters needed and then creates a disparity map. Disparity parameters are able to be changed in real time.
Description of disparity parameters:
https://vovkos.github.io/doxyrest-showcase/opencv/sphinx_rtd_theme/class_cv_StereoSGBM.html

Run using(w_runscript.py):
from camera_calibrate import StereoCalibration
cal = StereoCalibration('C:/Users/William/hello/ImagesD435_Urval/')

(Change path accordingly)

The images that you want to use to create the disparity map with need to be specified in the beginning of the code.

Close using q key.

Main source: https://github.com/markdtw/computer-vision-tasks

# RealsenseD435_Infrared_w_tabild.py
Snaps raw infrared images with the Intel Realsense D435 stereo camera with the laser turned off and saves them in separate folders LEFT and RIGHT.
