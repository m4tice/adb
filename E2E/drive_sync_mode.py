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
import blockweek_ad.ca_utils.dictionaries as dic
import blockweek_ad.ca_utils.tools as to
import blockweek_ad.ca_utils.sensor_utils as su

import time
import keyboard
import argparse
import numpy as np

from blockweek_ad.E2E.control import Controller
from blockweek_ad.ca_utils.tools import s2b
from tensorflow.keras.models import load_model
from queue import Queue
from queue import Empty

np.random.seed(2)

# -- VARIABLE INITIALIZATION --
town_dic = dic.town04
course_dict = dic.course1


def sensor_callback(sensor_data, sensor_queue, sensor_name):
    sensor_queue.put((sensor_data, sensor_name))


def game_loop(args):
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    # List initialization
    actor_list, sensor_list, features = [], [], []
    sensor_queue = Queue()

    try:
        if args.reload:
            print("Loading world...")
            world = client.load_world(town_dic['name'])
        else:
            world = client.get_world()

        settings = world.get_settings()
        settings.synchronous_mode = True
        world.apply_settings(settings)

        # Spectator
        spectator = world.get_spectator()
        transform = carla.Transform(carla.Location(x=town_dic['x'], y=town_dic['y'], z=town_dic['z']),
                                    carla.Rotation(pitch=town_dic['pitch'], yaw=town_dic['yaw']))
        spectator.set_transform(transform)

        # World setting
        world.set_weather(getattr(carla.WeatherParameters, town_dic['weather']))  # set weather
        blueprint_library = world.get_blueprint_library()  # get blueprint library

        # Vehicle
        vehicle_model = "model3"  # "model3"/"lincoln"/"mercedesccc"
        vehicle_color = dic.orange
        vehicle = su.spawn_vehicle(world, blueprint_library, vehicle_model, course_dict, vehicle_color)
        actor_list.append(vehicle)

        # Prediction Model
        print('- Loading model...')
        model_name = './models/model-t4-020-0.000022.h5'
        model = load_model(model_name)
        print("- Using model: {}".format(model_name))

        # Controller
        controller = Controller(vehicle, model, min_spd=2, max_spd=5, cam_set=dic.cam_rgb_set_2)

        # == Camera: RGB
        cam_rgb = su.spawn_rgb_camera(world, blueprint_library, dic.cam_rgb_set_2, vehicle)
        cam_rgb.listen(lambda data: sensor_callback(data, sensor_queue, 1))
        sensor_list.append(cam_rgb)

        # == Camera: spectator
        cam_spectate = su.spawn_rgb_camera(world, blueprint_library, dic.cam_spectate_1, vehicle)
        cam_spectate.listen(lambda data: sensor_callback(data, sensor_queue, 2))
        sensor_list.append(cam_spectate)

        # Sensor listening
        print("Stage: listening to sensor")
        while True:
            world.tick()
            try:
                for _ in range(len(sensor_list)):
                    s_frame = sensor_queue.get(True, 1.0)
                    if s_frame[1] == 1:
                        pass
                        controller.autodrive(s_frame[0], display_log=True, camera=args.cv_stream, pp1=True)
                    else:
                        if args.cv_stream:
                            su.live_cam(s_frame[0], dic.cam_spectate_1)
                        else:
                            pass

            # Stopping key
            except KeyboardInterrupt:
                print("Terminated")
                break

            except Empty:
                print("- Some of the sensor information is missed")

    finally:
        print("Finally...")

        # Switch back to synchronous mode
        settings.synchronous_mode = False
        world.apply_settings(settings)
        try:
            # Destroying actors
            print("Destroying {} actor(s)".format(len(actor_list)))
            client.apply_batch([carla.command.DestroyActor(x) for x in sensor_list])
            client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])

        except Exception as e:
            print("Final Exception: ", e)


def main():
    parser = argparse.ArgumentParser(description='Demonstration setup')
    parser.add_argument('-r', help='reload world',  dest='reload',    type=s2b, default='false')
    parser.add_argument('-o', help='opencv stream', dest='cv_stream', type=s2b, default='false')
    args = parser.parse_args()

    game_loop(args)
    time.sleep(0.5)


if __name__ == '__main__':
    main()
