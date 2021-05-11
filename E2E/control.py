import glob
import os
import sys

try:
    sys.path.append(glob.glob('../../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

sys.path.append(r"../../")
import blockweek_ad.ca_utils.training_util as tu

import math
import cv2
import numpy as np


class Controller:
    def __init__(self, vehicle, model, min_spd, max_spd, cam_set):
        print("<!> A Control Unit is created for:", vehicle.type_id)
        self.vehicle = vehicle
        self.model = model
        self.im_height = int(cam_set['image_size_y'])
        self.im_width = int(cam_set['image_size_x'])
        self.min_spd = min_spd
        self.max_spd = max_spd
        self.speed_limit = self.max_spd
        self.image = None
        self.steer = 0
        self.throttle = None
        self.brake = None
        self.level = None
        self.v = self.vehicle.get_velocity()
        self.speed = math.sqrt(self.v.x ** 2 + self.v.y ** 2 + self.v.z ** 2)

    # Build image from raw data
    def build_image(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (self.im_height, self.im_width, 4))
        array = array[:, :, :3]

        return array

    # Predict steering from image
    def predict_steering(self, image):
        self.image = self.build_image(image)

        main_array = self.image[:, :, ::-1]
        img = tu.preprocess1(main_array)
        img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))

        steer = self.model.predict(img)
        steer = float(steer[0][0])

        return steer

    # Estimate throttle and brake based on steer value
    def steering_to_throttle(self, steering):
        self.v = self.vehicle.get_velocity()
        self.speed = math.sqrt(self.v.x ** 2 + self.v.y ** 2 + self.v.z ** 2)

        if self.speed > self.max_spd:
            self.speed_limit = self.min_spd  # slow down
            throttle = 0.0
            brake = 1.0
        else:
            throttle = 1.0 - (steering ** 2) - (self.speed / self.speed_limit) ** 2
            self.speed_limit = self.max_spd
            brake = 0.0

        return throttle, brake

    def show_opencv_window(self, pp1=False):
        main_array = self.image
        if pp1:
            main_array = self.image[:, :, ::-1]
            main_array = tu.preprocess1(main_array)

        cv2.imshow("front_cam", main_array)
        cv2.waitKey(1)

    def autodrive(self, image, display_log=False, camera=False, pp1=False):
        self.steer = self.predict_steering(image)
        self.throttle, self.brake = self.steering_to_throttle(self.steer)
        self.vehicle.apply_control(carla.VehicleControl(throttle=self.throttle, steer=self.steer, brake=self.brake))

        # Display control values
        if display_log:
            print("THROTTLE: %6.3f :: STEER: %6.3f :: BRAKE: %6.3f :: SPD: %6.3f [m/s] :: SPD_LIMIT: %6.3f [m/s]" %
                  (self.throttle, self.steer, self.brake, self.speed, self.speed_limit))

        # Display camera view
        if camera:
            self.show_opencv_window(pp1=pp1)

