#!/usr/bin/env python3


import os

import cv2
import numpy as np
from yaml import load
from config_init import Client
frame_width=320
frame_height=240  

lane_detection_client = Client(False)

#calibration_img_files = glob.glob('/home/zlj/projects/udacity/advanced_lane_detection/peter-moran/highway-lane-tracker/data/camera_cal/*.jpg')
#lane_shape = [(584, 458), (701, 458), (295, 665), (1022, 665)]

#lane_shape = [(130, 15), (190, 15), (-180, 160), [320+180, 160]] view 12m,x 30deg
#lane_shape = [(60, 50), (260, 50), (-145, 160), [320+145, 160]]#view 3m,x 30deg
#lane_shape = [(30, 70), (290, 70), (-150, 160), [320+150, 160]]#view 2m,x 30deg
shangdi=200
xiadi=700
gao=100
kuan=320

source = np.float32([(frame_width/2-shangdi/2, frame_height-gao), (frame_width/2+shangdi/2, frame_height-gao), (frame_width/2-xiadi/2, frame_height), (frame_width/2+xiadi/2, frame_height)])
destination = np.float32([(frame_width/2-kuan/2, 0), (frame_width/2+kuan/2, 0),
                                  (frame_width/2-kuan/2, frame_height), (frame_width/2+kuan/2,frame_height)])
overhead_transform = cv2.getPerspectiveTransform(source, destination)
inverse_overhead_transform = cv2.getPerspectiveTransform(destination, source)
#config_path = os.path.join(os.path.dirname(__file__), '../cfg_file/calibration/LogitechC310/640_480/ost.yaml')
print (source)

config_path = os.path.join(os.path.dirname(__file__), '../cfg_file/calibration/LogitechC310/640_480/ost.yaml')


with open(config_path,'rb') as f:
    cont = f.read()

cf = load(cont)

camera_matrix=cf.get('camera_matrix')['data']
rows=cf.get('camera_matrix')['rows']
cols=cf.get('camera_matrix')['cols']
camera_matrix=np.reshape(camera_matrix, (rows, cols))
print (camera_matrix)

distortion_coefficients=cf.get('distortion_coefficients')['data']
rows=cf.get('distortion_coefficients')['rows']
cols=cf.get('distortion_coefficients')['cols']
distortion_coefficients=np.reshape(distortion_coefficients, (rows, cols))
print (distortion_coefficients)

print ('...waiting for message..' )

cap = cv2.VideoCapture(1)


while True: 
    #frame=cv2.imread("/home/nvidia/Desktop/test.png")
    ret, frame = cap.read()
    #print  (frame.shape)
    # show a frame

    processed_undistort = cv2.resize(frame,(640,480))
    processed_undistort=cv2.undistort(processed_undistort, camera_matrix, distortion_coefficients, None, camera_matrix)
    processed_undistort = cv2.resize(processed_undistort,(320,240))
    #processed_perspect=cv2.warpPerspective(processed_undistort, overhead_transform, dsize=(320, 240))
     
    scores = np.zeros(processed_undistort.shape[0:2]).astype('uint8')
    images = []
    image = processed_undistort / 255.0
    image = image.astype(np.float32)
    images.append(image)
    mask_prob = lane_detection_client.session.run(lane_detection_client.output_tensor, feed_dict={lane_detection_client.input_tensor: images})
    points_x = []
    points_y = []
    for i in range(0,lane_detection_client.img_h):
        points_x.append(i * 1.0)
        count=0
        for j in range(0, lane_detection_client.img_w):
            if mask_prob[0][i][j][0] < mask_prob[0][i][j][1]:
                count=count+1
                cv2.circle(scores, (j, i), 1, (255, 255, 255), 1)
                if count==lane_detection_client.bound_point_num:
                    break
        for j in range(lane_detection_client.img_w - 1, -1, -1):
            if mask_prob[0][i][j][0] < mask_prob[0][i][j][1]:
                count=count+1
                cv2.circle(scores, (j, i), 1, (255, 255, 255), 1)
                if count==lane_detection_client.bound_point_num * 2 :
                    break
    processed_frame, y_fit, x_fit_center, x_fit_center_derivative, status = lane_detection_client.lane_finder.viz_find_lines(scores)


    '''
    cv2.circle(processed_undistort, (np.int32(source[0][0]),np.int32(source[0][1])), 2, (0,0,255),2)
    cv2.circle(processed_undistort, (np.int32(source[1][0]),np.int32(source[1][1])), 2, (0,0,255),2)

    hengzuobiao1=50
    hengzuobiao2=265
    cv2.circle(processed_perspect, (hengzuobiao1,20), 2, (0,0,255),2)
    cv2.circle(processed_perspect, (hengzuobiao2,20), 2, (0,0,255),2)
    cv2.circle(processed_perspect, (hengzuobiao1,120), 2, (0,0,255),2)
    cv2.circle(processed_perspect, (hengzuobiao2,120), 2, (0,0,255),2)
    cv2.circle(processed_perspect, (hengzuobiao1,180), 2, (0,0,255),2)
    cv2.circle(processed_perspect, (hengzuobiao2,180), 2, (0,0,255),2)
    #cv2.imshow("capture", frame)
    #cv2.imshow("-capture", ~frame)
    cv2.imshow("processed_undistort", processed_undistort)    
    cv2.imshow("processed_perspect", processed_perspect)
    '''
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() 
