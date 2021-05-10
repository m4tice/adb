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

import cv2
import time
import math
import random

import numpy as np
import blockweek_ad.ca_utils.tools as to
import blockweek_ad.ca_utils.training_util as tu


# -- Spawn rgb camera --------
def spawn_rgb_camera(world, blueprint_library, dic_set, attached_actor):
    # get blueprint
    cam_rgb_bp = blueprint_library.find(dic_set['name'])

    # setting attributes
    if 'image_size_x' in dic_set:
        cam_rgb_bp.set_attribute('image_size_x', dic_set['image_size_x'])
    if 'image_size_y' in dic_set:
        cam_rgb_bp.set_attribute('image_size_y', dic_set['image_size_y'])
    if 'fov' in dic_set:
        cam_rgb_bp.set_attribute('fov', dic_set['fov'])
    if 'sensor_tick' in dic_set:
        cam_rgb_bp.set_attribute('sensor_tick', dic_set['sensor_tick'])

    # define spawning point
    cam_rgb_sp = carla.Transform(carla.Location(x=dic_set['loc_x'], z=dic_set['loc_z']),
                                 carla.Rotation(pitch=dic_set['rot_pitch']))

    # spawn
    cam_rgb = world.spawn_actor(cam_rgb_bp, cam_rgb_sp, attach_to=attached_actor)

    return cam_rgb


# -- Spawn vehicle --------
def spawn_vehicle(world, blueprint_library, model_name, course_dict=None, color=None):
    # get blueprint
    car_bp = blueprint_library.filter(model_name)[0]

    # set color
    if color is not None:
        car_bp.set_attribute('color', color)

    # set spawning point
    if course_dict is None:
        car_sp = random.choice(world.get_map().get_spawn_points())
    else:
        # only for town04
        try:
            course = to.load_course_data("../database/{}".format(course_dict['filename']))
        except FileNotFoundError:
            course = to.load_course_data("./database/{}".format(course_dict['filename']))

        car_sp = carla.Transform(carla.Location(x=course[0][0], y=course[0][1], z=0.5),
                                 carla.Rotation(yaw=course_dict['init_yaw']))

    # spawn
    vehicle = world.spawn_actor(car_bp, car_sp)
    print("Vehicle spawned at {}".format(course_dict['name']))

    return vehicle


# -- Passing red lights --------
def passing_trafficlight(vehicle):
    if vehicle.is_at_traffic_light():
        traffic_light = vehicle.get_traffic_light()
        if traffic_light.get_state() == carla.TrafficLightState.Red:
            # world.hud.notification("Traffic light changed! Good to go!")
            traffic_light.set_state(carla.TrafficLightState.Green)


# -- Speed estimation
def speed_estimation(vehicle):
    v = vehicle.get_velocity()
    ms = math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)
    kmh = int(3.6 * ms)
    print('Speed: %.4f (m/s) - %.4f (km/h)' % (ms, kmh))

    return ms, kmh


# -- Display image -----
def live_cam(image, dic_set, window_name=None, pp1=False):
    cam_height, cam_width = int(dic_set['image_size_y']), int(dic_set['image_size_x'])
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (cam_height, cam_width, 4))
    image = array[:, :, :3]

    if pp1:
        image = image[:, :, ::-1]
        image = tu.preprocess1(image)

    cv2.imshow(window_name, image)
    cv2.waitKey(1)


# -- Collect data for training --------
def record_data(image, world, vehicle, features, path):
    # passing red traffic lights
    passing_trafficlight(vehicle=vehicle)

    # get at traffic light state
    at_traffic_light = vehicle.is_at_traffic_light()

    # ger lane information
    waypoint = world.get_map().get_waypoint(vehicle.get_location(), project_to_road=True, lane_type=(
            carla.LaneType.Driving | carla.LaneType.Shoulder | carla.LaneType.Sidewalk))
    lane_type = waypoint.lane_type
    left_lane = waypoint.left_lane_marking.type
    right_lane = waypoint.right_lane_marking.type

    ms, kmh = speed_estimation(vehicle)

    # get control information
    control = vehicle.get_control()
    throttle, brake, steer = control.throttle, control.brake, control.steer

    # get location information
    location = vehicle.get_location()
    loc_x, loc_y = location.x, location.y

    # save images
    img_name = int(time.time() * 1000)
    image.save_to_disk("%s/%d.png" % (path, img_name))

    features.append(["IMG/{}.png".format(str(img_name)), loc_x, loc_y, at_traffic_light, left_lane, right_lane, throttle, brake, steer])
