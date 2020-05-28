import pyrealsense2 as rs
import numpy as np
import cv2
import os
from time import sleep


# Press Q to quit
ctx = rs.context() #Kod för att Stänga av laseremitter
devices = ctx.query_devices()
for dev in devices:
    sensors = dev.query_sensors()
for sensor in sensors:
    if sensor.is_depth_sensor():
        sensor.set_option(rs.option.emitter_enabled, 0) # Example to disable emitter

points = rs.points()
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
config.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 30)
profile = pipeline.start(config)

try:
    while True:
        frames = pipeline.wait_for_frames()
        nir_lf_frame = frames.get_infrared_frame(1)
        nir_rg_frame = frames.get_infrared_frame(2)
        if not nir_lf_frame or not nir_rg_frame:
            continue
        nir_lf_image = np.asanyarray(nir_lf_frame.get_data())
        nir_rg_image = np.asanyarray(nir_rg_frame.get_data())
        # horizontal stack
        image=np.hstack((nir_lf_image,nir_rg_image))
        cv2.namedWindow('NIR images (left, right)', cv2.WINDOW_AUTOSIZE)
        cv2.imwrite("D435.jpg", image)  #Spara bilden
        
        path_lf = 'C:/users/william/hello/'     # Path till folder med vänsterbilder
        path_rg = 'C:/users/william/hello/'     # Path till folder med högerbilder

        cv2.imwrite(os.path.join(path_lf, 'D435_Depth_Left.jpg'), nir_lf_image) # Spara vänster bild i spec. map
        cv2.imwrite(os.path.join(path_rg, 'D435_Depth_Right.jpg'), nir_rg_image) # Spara höger bild i spec. map

        #cv2.imwrite("D435_Left.jpg", nir_lf_image)  # Spara vänster bild
        #cv2.imwrite("D435_Right.jpg", nir_rg_image) # Spara höger bild
        #cv2.imshow('IR Example', image) #Visa bilden
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
finally:
    pipeline.stop()