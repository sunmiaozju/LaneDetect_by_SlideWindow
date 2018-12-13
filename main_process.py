# -*- encoding:utf-8 -*-
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
from tensorflow.python.platform import tf_logging as logging
from tensorflow.contrib.layers.python.layers import initializers
from find_lane import DashboardCamera, LaneFinder
from yaml import load


class LaneTracking:

    def __init__(self):
        self.img_w = 320
        self.img_h = 240
        self.model_source = "./model/"
        self.model_graph = "./model/model.ckpt.meta"
        self.outputTensorName = "ENet/logits_to_softmax:0"
        self.inputTensorName = "inputs_gate:0"
        self.bound_point_num = 5

        config_path = './cfg_file/config.yaml'
        # config_path = './cfg_file/config_unity3d_sim.yaml'
        with open(config_path, 'rb') as f:
            cont = f.read()
        self.cf = load(cont)

    def enet_model(self, images):
        ckpt = tf.train.get_checkpoint_state(self.model_source)
        saver = tf.train.import_meta_graph(self.model_graph)
        graph = tf.get_default_graph()
        input_tensor = graph.get_tensor_by_name(self.inputTensorName)
        output_tensor = graph.get_tensor_by_name(self.outputTensorName)
        session = tf.InteractiveSession()
        saver.restore(session, ckpt.model_checkpoint_path)
        mask_prob = session.run(output_tensor, feed_dict={input_tensor: images})
        return mask_prob

    def read_image(self):
        frame = cv2.imread('xuancheng.jpg')
        frame = cv2.resize(frame, (self.img_w, self.img_h))
        images = []
        image = frame / 255.0
        image = image.astype(np.float32)
        images.append(image)
        return images

    def detect_edge(self, mask_prob):
        points_x = []
        scores = np.zeros(mask_prob.shape[1:3]).astype('uint8')
        for i in range(0, self.img_h):
            points_x.append(i * 1.0)
            count = 0
            for j in range(0, self.img_w):
                if mask_prob[0][i][j][0] < mask_prob[0][i][j][1]:
                    count = count + 1
                    if j >= 10:
                        cv2.circle(scores, (j, i), 1, (255, 255, 255), 1)
                    if count == self.bound_point_num:
                        break
            for j in range(self.img_w - 1, -1, -1):
                if mask_prob[0][i][j][0] < mask_prob[0][i][j][1]:
                    count = count + 1
                    if self.img_w - j >= 10:
                        cv2.circle(scores, (j, i), 1, (255, 255, 255), 1)
                    if count == self.bound_point_num * 2:
                        break
        return scores

    def slide_windows(self, lane_edge):

        cf = self.cf
        # y轴像素对应实际距离
        scale_y_m_per_pix = cf.get('vision')['scale']['y_dis'] / cf.get('vision')['scale']['y_pix']
        # x轴像素对应实际距离
        scale_x_m_per_pix = cf.get('vision')['scale']['x_dis'] / cf.get('vision')['scale']['x_pix']
        scale = (scale_y_m_per_pix, scale_x_m_per_pix)

        # 计算ipm转换矩阵
        img_source_top = cf.get('vision')['img_source']['top_line']
        img_source_bottom = cf.get('vision')['img_source']['bottom_line']
        img_source_height = cf.get('vision')['img_source']['height']
        img_source = np.float32([((self.img_w - img_source_top) / 2, self.img_h - img_source_height),
                                 ((self.img_w + img_source_top) / 2, self.img_h - img_source_height),
                                 ((self.img_w - img_source_bottom) / 2, self.img_h),
                                 ((self.img_w + img_source_bottom) / 2, self.img_h)])

        img_destination_w = cf.get('vision')['img_destination']['width']
        img_destination_h = cf.get('vision')['img_destination']['height']
        img_destination = np.float32([((self.img_w - img_destination_w) / 2, self.img_h - img_destination_h),
                                      ((self.img_w + img_destination_w) / 2, self.img_h - img_destination_h),
                                      ((self.img_w - img_destination_w) / 2, self.img_h),
                                      ((self.img_w + img_destination_w) / 2, self.img_h)])

        window_shape_h = cf.get('vision')['lane_finder']['window_shape_h']
        window_shape_w = cf.get('vision')['lane_finder']['window_shape_w']
        search_margin = cf.get('vision')['lane_finder']['search_margin']
        max_frozen_dur = cf.get('vision')['lane_finder']['max_frozen_dur']

        camera = DashboardCamera(scale_correction=scale,
                                 h=self.img_h,
                                 w=self.img_w,
                                 source=img_source,
                                 destination=img_destination)

        lane_finder = LaneFinder(camera,
                                 (window_shape_h, window_shape_w),
                                 search_margin,
                                 max_frozen_dur)

        return lane_finder.viz_find_lines(lane_edge)

    def lane_process(self):
        images = self.read_image()
        mask_prob = self.enet_model(images)
        lane_edge = self.detect_edge(mask_prob)
        cv2.imwrite('./pic_watch/lane_edg.png', lane_edge)
        cv2.imwrite('./pic_watch/orign.png', images[0] * 255.0)
        y_fit, x_fit_center, x_fit_center_derivative, status = self.slide_windows(lane_edge)

        # 1, vision info
        derivat_tan = x_fit_center_derivative[len(
            x_fit_center) // 2] * self.lane_detection_client.camera.x_m_per_pix / self.lane_detection_client.camera.y_m_per_pix
        eoa = np.arctan2(derivat_tan, 1) / np.pi * 180
        x_pix = self.lane_detection_client.img_w // 2 - (x_fit_center[len(x_fit_center) // 2])
        eod = 1.0 * x_pix * self.lane_detection_client.camera.x_m_per_pix

        # 2, fuzzy convert process
        cte = self.lane_detection_client.fuzzyconvert.cal_cte_level(eoa, eod)

        # 3, control process
        self.lane_detection_client.pid.UpdateError(cte)
        steer_value = self.lane_detection_client.pid.TotalError()
        # 4, send ecu

        if self.is_in_task.is_set():
            ecu = ECU()
            ecu.motor = self.speed
            ecu.servo = steer_value
            ecu.shift = self.direction
            self.ecu_pub.publish(ecu)


if __name__ == '__main__':
    laneClient = LaneTracking()
    laneClient.lane_process()
