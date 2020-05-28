import numpy as np
import cv2
import glob
import argparse
import sys
import pdb
import os
from matplotlib import pyplot as plt


class StereoCalibration(object):
    def __init__(self, filepath):
        # termination criteria
        self.criteria = (cv2.TERM_CRITERIA_EPS +
                         cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.criteria_cal = (cv2.TERM_CRITERIA_EPS +
                             cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(7,5,0)
        self.objp = np.zeros((9*7, 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:9, 0:7].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        self.objpoints = []  # 3d point in real world space
        self.imgpoints_l = []  # 2d points in image plane.
        self.imgpoints_r = []  # 2d points in image plane.

        self.cal_path = filepath
        self.read_images(self.cal_path)

    def read_images(self, cal_path):
        images_right = glob.glob(cal_path + 'RIGHT/*.JPG')
        images_left = glob.glob(cal_path + 'LEFT/*.JPG')
        images_left.sort()
        images_right.sort()

        for i, fname in enumerate(images_right):
            img_l = cv2.imread(images_left[i])
            img_r = cv2.imread(images_right[i])

            gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret_l, corners_l = cv2.findChessboardCorners(gray_l, (9, 7), None)
            ret_r, corners_r = cv2.findChessboardCorners(gray_r, (9, 7), None)

            # If found, add object points, image points (after refining them)
            self.objpoints.append(self.objp)

            if ret_l is True:
                rt = cv2.cornerSubPix(gray_l, corners_l, (11, 11),
                                      (-1, -1), self.criteria)
                self.imgpoints_l.append(corners_l)

                # Draw and display the corners
                ret_l = cv2.drawChessboardCorners(img_l, (9, 7),
                                                  corners_l, ret_l)
                cv2.imshow(images_left[i], img_l)
                cv2.waitKey(500)

            if ret_r is True:
                rt = cv2.cornerSubPix(gray_r, corners_r, (11, 11),
                                      (-1, -1), self.criteria)
                self.imgpoints_r.append(corners_r)

                # Draw and display the corners
                ret_r = cv2.drawChessboardCorners(img_r, (9, 7),
                                                  corners_r, ret_r)
                cv2.imshow(images_right[i], img_r)
                cv2.waitKey(500)
            img_shape = gray_l.shape[::-1]

        rt, self.M1, self.d1, self.r1, self.t1 = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_l, img_shape, None, None)
        rt, self.M2, self.d2, self.r2, self.t2 = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_r, img_shape, None, None)

        self.camera_model = self.stereo_calibrate(img_shape)

    def stereo_calibrate(self, dims):
        flags = 0
        flags |= cv2.CALIB_FIX_INTRINSIC
        # flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
        flags |= cv2.CALIB_USE_INTRINSIC_GUESS
        flags |= cv2.CALIB_FIX_FOCAL_LENGTH
        # flags |= cv2.CALIB_FIX_ASPECT_RATIO
        flags |= cv2.CALIB_ZERO_TANGENT_DIST
        # flags |= cv2.CALIB_RATIONAL_MODEL
        # flags |= cv2.CALIB_SAME_FOCAL_LENGTH
        # flags |= cv2.CALIB_FIX_K3
        # flags |= cv2.CALIB_FIX_K4
        # flags |= cv2.CALIB_FIX_K5

        stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER +
                                cv2.TERM_CRITERIA_EPS, 100, 1e-5)
        ret, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate(
            self.objpoints, self.imgpoints_l,
            self.imgpoints_r, self.M1, self.d1, self.M2,
            self.d2, dims,
            criteria=stereocalib_criteria, flags=flags)
        ################################################3
        print ('rectifying...')
            # Step 3: Stereo rectification
        if len(sys.argv) == 2:
            rectify_scale = float(sys.argv[1])
        else:
            rectify_scale = 0
            R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(self.M1, self.d1, self.M2, self.d2, dims, R, T, alpha=rectify_scale)
        ######### undistort
        #cam_m=M1#np.array([[2590.2742, 0, 997.78192],[0, 2582.9724, 509.76907],[0, 0, 1]])
        #([f_x 0 c_x],[0 f_y c_y],[0 0 1])
        #dist_c=d1#np.array([0.088253155, 0.96952456, 0.0033740622, -0.00064934365, -6.0030732])
        #([distk1 distk2 distp1 distp2 dist])
        map1_left,map2_left=cv2.initUndistortRectifyMap(M1, d1, None, None, dims, cv2.CV_32FC1)
        map1_right,map2_right=cv2.initUndistortRectifyMap(M2, d2, None, None, dims, cv2.CV_32FC1)
        ######### use the rectified data to do remap
        imgL = cv2.imread('D435_Depth_Left.jpg',1)
        imgR = cv2.imread('D435_Depth_Right.jpg',1)
        Left_img_remap = cv2.remap(imgL, map1_left, map2_left, cv2.INTER_LINEAR)
        Right_img_remap = cv2.remap(imgR, map1_right, map2_right, cv2.INTER_LINEAR)
        ################################################ Stereo compute
        def minDispsCallBack(x):
            pass
        def numDispsCallBack(x):
            pass
        def bSizeCallBack(x):
            pass
        def wSizeCallBack(x):
            pass
        def disp12CallBack(x):
            pass
        def uniqCallBack(x):
            pass
        def spWCallBack(x):
            pass
        def spRCallBack(x):
            pass
        ################################################# TUNE DISPARITY
        def tuneDisparity(lframe, rframe, l_maps, r_maps):
                
            ## use the rectified data to do remap on webcams
            #lframe_remap = cv2.remap(lframe, l_maps[0], l_maps[1], cv2.INTER_LINEAR)
            #rframe_remap = cv2.remap(rframe, r_maps[0], r_maps[1], cv2.INTER_LINEAR)

            minDisp = cv2.getTrackbarPos('minDisparity', 'disparity_parameters')
            numDisp = cv2.getTrackbarPos('numDisparities', 'disparity_parameters') * 16
            blockSize = cv2.getTrackbarPos('blockSize', 'disparity_parameters')
            blockSize = blockSize > 1 and blockSize or 2
            SADWindowSize = cv2.getTrackbarPos('SADWindowSize', 'disparity_parameters')
            P1 = 8*3*SADWindowSize**2
            P2 = 32*3*SADWindowSize**2
            disp12MaxDiff = cv2.getTrackbarPos('disp12MaxDiff', 'disparity_parameters')
            uniquenessRatio = cv2.getTrackbarPos('uniquenessRatio', 'disparity_parameters')
            speckleWindowSize = cv2.getTrackbarPos('speckeWindowSize', 'disparity_parameters')
            speckleRange = cv2.getTrackbarPos('speckleRange', 'disparity_parameters')

            stereo = cv2.StereoSGBM_create(\
                    minDisparity=minDisp,
                    numDisparities=numDisp,
                    blockSize=blockSize,
                    P1=P1,
                    P2=P2,
                    disp12MaxDiff=disp12MaxDiff,
                    uniquenessRatio=uniquenessRatio,
                    speckleWindowSize=speckleWindowSize,
                    speckleRange=speckleRange)

            disparity = stereo.compute(l_maps, r_maps).astype(np.float32) / 16.0
            optimal_disparity = (disparity - minDisp) / numDisp
            cv2.imshow('disparaty_map', optimal_disparity)
        ################################################################

        cv2.namedWindow('disparity_parameters')
        text = np.zeros((5, 500), dtype=np.uint8)
        cv2.imshow('disparity_parameters', text)
        cv2.createTrackbar('minDisparity', 'disparity_parameters', 1, 100, minDispsCallBack)
        cv2.createTrackbar('numDisparities', 'disparity_parameters', 3, 20, numDispsCallBack)    # divisible by 16
        cv2.createTrackbar('blockSize', 'disparity_parameters', 2, 30, bSizeCallBack)            # odd number, 1 < 3 < blockSize < 11
        cv2.createTrackbar('SADWindowSize', 'disparity_parameters', 4, 30, wSizeCallBack)        
        #cv2.createTrackbar('P1', 'disparity_parameters', 1, 1, p1SizeCallBack)
        #cv2.createTrackbar('P2', 'disparity_parameters', 1, 1, p2SizeCallBack)
        cv2.createTrackbar('disp12MaxDiff', 'disparity_parameters', 0, 30, disp12CallBack)
        cv2.createTrackbar('uniquenessRatio', 'disparity_parameters', 5, 30, uniqCallBack)
        cv2.createTrackbar('speckleWindowSize', 'disparity_parameters', 100, 200, spWCallBack)   # 55 < speckleWindow < 200
        cv2.createTrackbar('speckleRange', 'disparity_parameters', 2, 32, spRCallBack)           # 1 <= speckleRange <= 2
        #Info om parametrar ovan https://vovkos.github.io/doxyrest-showcase/opencv/sphinx_rtd_theme/class_cv_StereoSGBM.html
        minDisp = cv2.getTrackbarPos('minDisparity', 'disparity_parameters')
        numDisp = cv2.getTrackbarPos('numDisparities', 'disparity_parameters') * 16
        blockSize = cv2.getTrackbarPos('blockSize', 'disparity_parameters')
        blockSize = blockSize > 1 and blockSize or 2
        SADWindowSize = cv2.getTrackbarPos('SADWindowSize', 'disparity_parameters')
        P1 = 8*3*SADWindowSize**2
        P2 = 32*3*SADWindowSize**2
        disp12MaxDiff = cv2.getTrackbarPos('disp12MaxDiff', 'disparity_parameters')
        uniquenessRatio = cv2.getTrackbarPos('uniquenessRatio', 'disparity_parameters')
        speckleWindowSize = cv2.getTrackbarPos('speckeWindowSize', 'disparity_parameters')
        speckleRange = cv2.getTrackbarPos('speckleRange', 'disparity_parameters')

        while(True):
            #focal_length_x = M1[0][0]
            # show disparity map and tune the paramters in real-time
            tuneDisparity(imgL, imgR, Left_img_remap, Right_img_remap)

            key = cv2.waitKey(5)&0xFF
            if key == 27 or key == ord('q'):
                print('bye')
                break
        
#        disparity = stereo.compute(Left_img_remap, Right_img_remap).astype(np.float32) / 16.0
#        optimal_disparity = (disparity - minDisp) / numDisp
#        cv2.imshow('disparaty_map', optimal_disparity)
#        path_disp = 'C:/users/william/hello/'     # Path till folder med högerbilder
#        cv2.imwrite(os.path.join(path_disp, 'Disparity_map.jpg'), optimal_disparity) # Spara disparity map

        ################################################3
        print('~~~~~~~~~~~~~~~~~~~ Stereo Calibrate ~~~~~~~~~~~~~~~~~~~')
        print('Intrinsic_mtx_1:', M1)
        print('dist_1:', d1)
        print('Intrinsic_mtx_2:', M2)
        print('dist_2:', d2)
        print('R:', R)
        print('T:', T)                      #STEREO CALIBRATION PARAMETERS
        print('E:', E)
        print('F:', F)
        ###############################################
        print('~~~~~~~~~~~~~~~~~~~ Stereo Rectify ~~~~~~~~~~~~~~~~~~~')
        print('Cam 1 3x3 recti. transf. (rot. matrix):', R1)
        print('Cam 2 3x3 recti. transf. (rot. matrix):', R2)
        print('Cam 1 3x4 proj. matrix in new (rectified) coord. sys.:', P1)
        print('Cam 2 3x4 proj. matrix in new (rectified) coord. sys.:', P2)
        print('4x4 disparity-to-depth map. matrix:', Q)
        print('validPixROI1:', roi1)        # STEREO REXTIFICATION PARAMETERS
        print('validPixROI2:', roi2)
        ################################################
        print('~~~~~~~~~~~~~~~~~~~ initUndistortRectifyMap ... ')
        #print('Cam 1 Map1:', map1)
        #print('Cam 1 Map2:', map2)
        print('~~~~~~~~~~~~~~~~~~~ Remaping ... ')
        #
        print('~~~~~~~~~~~~~~~~~~~ Computing Stereo ... ')
        print('~~~~~~~~~~~~~~~~~~~ Creating disparity map ... ')
        # Step 2.5: Save the calibration stats to disk for future use
        with open('depthdata.txt', 'wb') as f:
            #f.write("cameraMatrix1:\n")
            np.savetxt(f, M1)
            #f.write("distCoeffs1:\n")
            np.savetxt(f, d1)
            #f.write("cameraMatrix2:\n")
            np.savetxt(f, M2)
            #f.write("distCoeffs2:\n")
            np.savetxt(f, d2)
            #f.write("R:\n")
            np.savetxt(f, R)
            #f.write("T:\n")
            np.savetxt(f, T)
            #f.write("E:\n")
            np.savetxt(f, E)
            #f.write("F:\n")
            np.savetxt(f, F)

        print('')

        camera_model = dict([('M1', M1), ('M2', M2), ('dist1', d1),
                            ('dist2', d2), ('rvecs1', self.r1),
                            ('rvecs2', self.r2), ('R', R), ('T', T),
                            ('E', E), ('F', F)])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return camera_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filepath', help='String Filepath')
    args = parser.parse_args()
    cal_data = StereoCalibration(args.filepath)
