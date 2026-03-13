# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# MediaPipe 0.10.31+ 移除了 solutions 模块，手部关键点连接关系（用于 OpenCV 绘制）
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),   # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),   # index
    (0, 9), (9, 10), (10, 11), (11, 12),  # middle
    (0, 13), (13, 14), (14, 15), (15, 16),  # ring
    (0, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (5, 9), (9, 13), (13, 17),  # palm
)
# import pyrealsense2 as rs
import cv2
# from geort.mocap.camera.realsense import RealSenseCamera
from geort.mocap.camera.webcam import WebcamCamera

class MediaPipeHandProcessor:
    def __init__(self):
        # we will need to track the orientation, and do some low pass filtering.
        self.last_rotation = None

    def ema_rotation_matrix(self, R1, R2, alpha):
        from scipy.spatial.transform import Rotation as R
        # Convert rotation matrices to quaternions
        q1 = R.from_matrix(R1).as_quat()
        q2 = R.from_matrix(R2).as_quat()
        
        # Create a slerp object and interpolate
        interpolated = self.slerp(q1, q2, alpha)
        
        # Convert back to a rotation matrix
        R_interpolated = R.from_quat(interpolated).as_matrix()
        return R_interpolated

    def slerp(self, q1, q2, t):
        q1 = np.array(q1)
        q2 = np.array(q2)
        dot_product = np.dot(q1, q2)
        if dot_product < 0.0:
            q1 = -q1
            dot_product = -dot_product

        if dot_product > 0.9995:  # Quaternions are close, use linear interpolation
            result = (1.0 - t) * q1 + t * q2
            return result / np.linalg.norm(result)

        theta_0 = np.arccos(dot_product)  # Angle between quaternions
        sin_theta_0 = np.sin(theta_0)      # Sine of the angle
        theta = theta_0 * t                # Angle for the interpolation
        sin_theta = np.sin(theta)          # Sine of the angle for the interpolation
        sin_theta_1 = np.sin(theta_0 - theta)  # Sine of the remaining angle

        q_interpolated = (sin_theta_1 * q1 + sin_theta * q2) / sin_theta_0
        return q_interpolated / np.linalg.norm(q_interpolated)

    def forward(self, hand_detection_result, apply_ema=False):
        # fine, apply_ema seems futile for MediaPipe.
        z_axis = hand_detection_result[9] - hand_detection_result[0]
        z_axis = z_axis / np.linalg.norm(z_axis)
        y_axis_aux = hand_detection_result[5] - hand_detection_result[13]
        y_axis_aux = y_axis_aux / np.linalg.norm(y_axis_aux)

        x_axis = np.cross(y_axis_aux, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)

        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)

        tranlation_base = hand_detection_result[0]
        rotation_base = np.array([x_axis, y_axis, z_axis]).transpose()

        if apply_ema:
            if self.last_rotation is not None:
                rotation_base = self.ema_rotation_matrix(self.last_rotation, rotation_base, 0.75)
            self.last_rotation = rotation_base
        
        transform = np.eye(4)
        transform[:3, :3] = rotation_base
        transform[:3, 3] = tranlation_base
        
        transform_inv = np.linalg.inv(transform)
        hand_detection_result_np = np.array(hand_detection_result)
        hand_detection_result_np = np.concatenate((np.array(hand_detection_result_np), np.ones((21, 1))), axis=-1)

        hand_detection_result_np = hand_detection_result_np @ transform_inv.transpose()
         
        return hand_detection_result_np


class MediaPipeHandDetector:
    MARGIN = 10  # pixels
    FONT_SIZE = 1
    FONT_THICKNESS = 1
    HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

    def __init__(self):
        base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
        options = vision.HandLandmarkerOptions(base_options=base_options,
                                            num_hands=2)
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.processor = MediaPipeHandProcessor()

        return

    def _draw_hand_landmarks_cv2(self, image, hand_landmarks, height, width):
        """使用 OpenCV 绘制手部关键点与连线，兼容新版 MediaPipe（无 solutions 模块）。"""
        # 绘制连线
        for start_idx, end_idx in HAND_CONNECTIONS:
            if start_idx < len(hand_landmarks) and end_idx < len(hand_landmarks):
                start = hand_landmarks[start_idx]
                end = hand_landmarks[end_idx]
                pt1 = (int(start.x * width), int(start.y * height))
                pt2 = (int(end.x * width), int(end.y * height))
                cv2.line(image, pt1, pt2, (0, 255, 0), 1, cv2.LINE_AA)
        # 绘制关键点
        for landmark in hand_landmarks:
            cx, cy = int(landmark.x * width), int(landmark.y * height)
            cv2.circle(image, (cx, cy), 2, (0, 205, 88), -1, cv2.LINE_AA)

    def draw_landmarks_on_image(self, rgb_image, detection_result):
        hand_landmarks_list = detection_result.hand_landmarks
        handedness_list = detection_result.handedness
        annotated_image = np.copy(rgb_image)
        height, width, _ = annotated_image.shape

        for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]
            handedness = handedness_list[idx]

            self._draw_hand_landmarks_cv2(
                annotated_image, hand_landmarks, height, width
            )

            x_coordinates = [landmark.x for landmark in hand_landmarks]
            y_coordinates = [landmark.y for landmark in hand_landmarks]
            text_x = int(min(x_coordinates) * width)
            text_y = int(min(y_coordinates) * height) - MediaPipeHandDetector.MARGIN

            category_name = (
                handedness[0].category_name
                if hasattr(handedness[0], "category_name")
                else str(handedness[0])
            )
            cv2.putText(
                annotated_image,
                category_name,
                (text_x, text_y),
                cv2.FONT_HERSHEY_DUPLEX,
                MediaPipeHandDetector.FONT_SIZE,
                MediaPipeHandDetector.HANDEDNESS_TEXT_COLOR,
                MediaPipeHandDetector.FONT_THICKNESS,
                cv2.LINE_AA,
            )

        return annotated_image

    def numpy_to_mp_image(self, image_np):
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        # Create a MediaPipe image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_np)
        return mp_image


    def detect(self, rgb_image):
        detection_result = self.detector.detect(self.numpy_to_mp_image(rgb_image))

        hand_landmarks = detection_result.hand_landmarks
        hand_world_landmarks = detection_result.hand_world_landmarks

        detected = False
        coordinates = []
        world_coordinates = []
        canonical_coordinates = []

        if len(hand_landmarks) > 0:
            hand_landmarks = hand_landmarks[0]
            for landmark in hand_landmarks:
                coordinates.append([landmark.x, landmark.y, landmark.z])

            detected = True
            
        if len(hand_world_landmarks) > 0:
            world_landmarks = hand_world_landmarks[0]
            for landmark in world_landmarks:
                world_coordinates.append([landmark.x, landmark.y, landmark.z])
        annotated_image = self.draw_landmarks_on_image(rgb_image, detection_result)
        
        if detected:
            canonical_coordinates = self.processor.forward(np.array(coordinates))[..., :3]

        
        # print(annotated_image.shape)
        return {
            'detected': detected, 
            "annotated_img": annotated_image, 
            "coordinates": np.array(coordinates),
            "canonical_coordinates": canonical_coordinates,
            "world_coordinates": np.array(world_coordinates)
        }

class MediaPipeMocap:
    def __init__(self):
        self.camera = WebcamCamera()
        self.detector = MediaPipeHandDetector()
        self.status = 'idle'

    def get(self):
        # Run the mocap system.
        rgb = self.camera.get_frame()["rgb"]
        result = self.detector.detect(rgb)
        
        # Show the live detection.
        cv2.imshow("detection", result["annotated_img"])
          
        # Keyboard Control.
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            self.status = 'quit' 
        elif key == ord('s'):
            self.status = 'recording'
        elif key == ord('e'):
            self.record = 'idle'

        detection = None 
        if result["detected"]:
            detection = result["canonical_coordinates"]
        return {'status': self.status, "result": detection}


if __name__ == '__main__':
    # A Naive Mocap System
    # - DO NOT use this for arm-hand teleop! 
    # -- very unstable during wrist movement.
    # -- significant hand scale shift during deployment, even optimization cannot save it.


    # Usage:
    # Step 1. Execute this script. 
    #         $ python ./geort/mocap/mediapipe_mocap.py --name [YOUR_DATASET_NAME]
    # Step 2. Set the window focus on the pop-up camera view.
    # Step 3. Press "s" to start recording, "e" to pause recording, and "q" to finish.
    #         Hint: move your hand to the center of camera view and press "s" to start.
    #               you will see the terminal start to print current status.
    # Step 4. Now you can train your GeoRT model with your collected [YOUR_DATASET_NAME] data.
    
    import argparse
    import geort
    

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='human1', type=str) # the data package name.
    args = parser.parse_args()
    
    dataset_name = args.name 

    # Mocap System.    
    mocap = MediaPipeMocap()
   
    # Data Collection Loop.
    all_results = []

    while True:
        result = mocap.get()

        if result['status'] == 'recording' and result["result"] is not None:
            all_results.append(result["result"])
            print("Data collected:", len(all_results))

        if result['status'] == 'quit':
            break 

    # Save!
    save_path = geort.save_human_data(np.array(all_results), args.name)
    print("Data saved to", save_path)
