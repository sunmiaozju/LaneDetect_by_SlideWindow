#!/usr/bin/python
# -*-encoding:utf-8-*-
import contextlib
import mmap
import time
import cv2
import numpy as np
import rospy
import threading
from can_msgs.msg import ECU
from tf2_msgs.msg import TFMessage
from track_msgs.msg import LaneSection, TaskShutdown, SectionTaskState
from monitor_msgs.msg import ModuleHealth, ModuleCheck
from usb_camera.msg import Picture
from config_init import Client
import tensorflow as tf
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
from tensorflow.python.platform import tf_logging as logging
from tensorflow.contrib.layers.python.layers import initializers


class LaneTracking:
    def __init__(self):
        self.module_name = "enet_lane_tracking"
        rospy.init_node(self.module_name)
        # What we do during shutdown
        self.lane_detection_client = Client()
        self.speed = 0
        self.direction = 0
        self.task = None
        self.task_state = None
        self.is_in_task = threading.Event()
        self.shm_filename = rospy.get_param('~shm_filename')
        self.last_left_bound = []
        self.last_right_bound = []
        rospy.Subscriber("ModuleCheck", ModuleCheck, self.module_check_cb, queue_size=10)
        rospy.Subscriber("Picture", Picture, self.picture_cb, queue_size=10)
        rospy.Subscriber("LaneSection", LaneSection, self.start_lane_cb, queue_size=10)
        rospy.Subscriber("TaskShutdown", TaskShutdown, self.stop_lane_cb, queue_size=10)
        rospy.Subscriber("tf", TFMessage, self.on_position, queue_size=10)
        self.task_reporter = rospy.Publisher("SectionTaskState", SectionTaskState, queue_size=10)
        self.module_reporter = rospy.Publisher("ModuleHealth", ModuleHealth, queue_size=10)
        self.ecu_pub = rospy.Publisher('ecu', ECU, queue_size=100)
        self.state = ModuleHealth.MODULE_OK
        self.moduleType = ModuleHealth.MODULE_TYPE_GENERAL

    def module_check_cb(self, msg):
        """
        :param msg:
        :type msg: ModuleCheck
        :return:
        """
        response = ModuleHealth()
        response.requestId = msg.requestId
        response.state = self.state
        response.moduleName = rospy.get_name()
        response.moduleType = self.moduleType
        self.module_reporter.publish(response)

    def on_position(self, tf_message):
        """
        according tf_message, judge whether the current lane following is finished.
        :param tf_message:
        :type tf_message TFMessage
        :return:
        """
        if self.task and self.task.goal.name == tf_message.transforms[0].child_frame_id:
            self.is_in_task.clear()
            self.task = None
            self.task_state.header.stamp = rospy.Time.now()
            self.task_state.percentage = 100
            self.task_state.state = 0x08  # finished
            self.task_reporter.publish(self.task_state)
            self.task_state = None

    def start_lane_cb(self, task):
        self.task = task
        self.speed = task.speed
        self.direction = task.direction
        self.task_state = SectionTaskState()
        self.task_state.sectionId = task.sectionId
        self.task_state.percentage = 0
        self.task_state.state = 0x03  # running
        self.task_state.header.stamp = rospy.Time.now()
        self.is_in_task.set()

    def stop_lane_cb(self, stop_msg):
        if self.task and stop_msg.sectionId == self.task.sectionId:
            rospy.loginfo("stop_lane_cb")
            self.is_in_task.clear()
            self.task = None
            self.task_state.header.stamp = rospy.Time.now()
            self.task_state.state = 0x31  # killed
            self.task_reporter.publish(self.task_state)
            self.task_state = None
            ecu = ECU()
            ecu.shift = 0
            ecu.servo = 0
            ecu.motor = 0
            self.ecu_pub.publish(ecu)

    def report_task(self):
        self.task_reporter.publish(self.task)

    def is_valid_jpg(self, jpg_file):
        with open(jpg_file, 'rb') as f:
            f.seek(-2, 2)
            return f.read() == '\xff\xd9'

    def picture_cb(self, message):
        """
        :param message:
        :type message Picture
        :return:
        """
        filter_flag = 0
        if len(self.last_left_bound) > 0 and len(self.last_right_bound) > 0:
            filter_flag = 1

        if self.is_in_task.is_set():
            if self.shm_filename != message.dev_name:  # device /dev/video1, shm_filename : /dev/shm/ss_file_name
                return
            mat = None
            try:
                with open(message.shm_filename, "r+") as f:
                    with contextlib.closing(mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)) as m:
                        m.seek(message.offset)
                        mat = m.read(message.block_size)

                        # print self.is_valid_jpg(m)

            except Exception, e:
                return
            if mat:
                nparr = np.fromstring(mat, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                # try to write to
                cv2.imwrite('/tmp/temp.jpg', frame)
                if not self.is_valid_jpg('/tmp/temp.jpg'):
                    self.state = ModuleHealth.MODULE_WARNING
                    return
                frame = cv2.undistort(frame, self.lane_detection_client.camera_matrix,
                                      self.lane_detection_client.distortion_coefficients, None,
                                      self.lane_detection_client.camera_matrix)
                frame = cv2.resize(frame, (self.lane_detection_client.img_w, self.lane_detection_client.img_h))
                images = []
                image = frame / 255.0
                image = image.astype(np.float32)
                images.append(image)
                mask_prob = self.lane_detection_client.session.run(self.lane_detection_client.output_tensor, feed_dict={
                    self.lane_detection_client.input_tensor: images})

                points_x = []
                points_y = []
                for i in range(int(self.lane_detection_client.img_h * (1 - self.lane_detection_client.area_blocks)),
                               self.lane_detection_client.img_h):
                    points_x.append(i * 1.0)
                    temp_y = 0
                    for j in range(0, self.lane_detection_client.img_w):
                        if mask_prob[0][i][j][0] < mask_prob[0][i][j][1]:
                            ratio = self.lane_detection_client.filter_ratio
                            index = i - int(
                                self.lane_detection_client.img_h * (1 - self.lane_detection_client.area_blocks))
                            if filter_flag == 0:
                                self.last_left_bound.append(j)
                            else:
                                self.last_left_bound[index] = self.last_left_bound[index] * ratio + j * (1 - ratio)
                            temp_y = temp_y + self.last_left_bound[index]
                            cv2.circle(frame, (int(self.last_left_bound[index]), i), 3, (0, 0, 255), 1)
                            break
                    for j in range(self.lane_detection_client.img_w - 1, -1, -1):
                        if mask_prob[0][i][j][0] < mask_prob[0][i][j][1]:
                            ratio = self.lane_detection_client.filter_ratio
                            index = i - int(
                                self.lane_detection_client.img_h * (1 - self.lane_detection_client.area_blocks))
                            if filter_flag == 0:
                                self.last_right_bound.append(j)
                            else:
                                self.last_right_bound[index] = self.last_right_bound[index] * ratio + j * (1 - ratio)
                            temp_y = temp_y + self.last_right_bound[index]
                            cv2.circle(frame, (int(self.last_right_bound[index]), i), 3, (0, 0, 255), 1)
                            break
                    points_y.append(temp_y / 2.0)
                fit_vals = dict()

                x = np.array(points_x)
                y = np.array(points_y)
                fit_cr = np.polyfit(x, y, 2)
                fit_vals['a'] = fit_cr[0]
                fit_vals['b'] = fit_cr[1]
                fit_vals['x0'] = fit_cr[2]

                for i in range(int(self.lane_detection_client.img_h * (1 - self.lane_detection_client.area_blocks)),
                               self.lane_detection_client.img_h):
                    cv2.circle(frame, (int(fit_vals['a'] * i * i + fit_vals['b'] * i + fit_vals['x0']), i), 1,
                               (0, 255, 0), 1)

                cv2.imshow("result", frame)
                cv2.waitKey(1)
                # 1, vision info
                y_pix = (self.lane_detection_client.img_h * (
                        1 - self.lane_detection_client.area_blocks) + self.lane_detection_client.img_h) / 2.0
                error_dis = self.lane_detection_client.img_w // 2 - (
                        fit_vals['a'] * y_pix * y_pix + fit_vals['b'] * y_pix + fit_vals['x0'])
                tan_value = 2.0 * fit_vals['a'] * y_pix + fit_vals['b']
                error_angle = np.arctan2(tan_value, 1) / np.pi * 180

                # 2, fuzzy convert process
                cte = self.lane_detection_client.fuzzyconvert.cal_cte_level(error_angle, error_dis)

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

    def shutdown(self):
        """
        :return:
        """
        if self.task:
            self.task = None
            ecu = ECU()
            ecu.servo = 0
            ecu.motor = 0
            ecu.shift = 0
            self.ecu_pub.publish(ecu)
        report = ModuleHealth()
        report.requestId = -1
        report.state = ModuleHealth.MODULE_ERROR
        report.moduleType = self.moduleType
        report.moduleName = rospy.get_name()
        self.module_reporter.publish(report)


if __name__ == '__main__':
    tracker = None
    try:
        tracker = LaneTracking()
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down vision node.")
    finally:
        if tracker:
            tracker.shutdown()
