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
import argparse
import keyboard
import numpy as np

from queue import Queue
from queue import Empty
from blockweek_ad.ca_utils.tools import s2b

np.random.seed(2)

# -- VARIABLE INITIALIZATION --
town_dic = dic.town04
course_dict = dic.course1
sd = 65.0

# -- PATHS --
img_dir = "./collected_data/IMG"
csv_file = "./collected_data/driving_log.csv"


def sensor_callback(sensor_data, sensor_queue, sensor_name):
    sensor_queue.put((sensor_data, sensor_name))


def game_loop(args, reload=True, nos=300):
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
        traffic_manager = client.get_trafficmanager()
        # traffic_manager.global_percentage_speed_difference(sd)

        # Vehicle
        vehicle_model = "model3"  # "model3"/"lincoln"/"mercedesccc"
        vehicle_color = dic.petronas_color
        vehicle = su.spawn_vehicle(world, blueprint_library, vehicle_model, course_dict=course_dict, color=vehicle_color)
        actor_list.append(vehicle)

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
        skip_lim = 30  # wait for collision at spawn to pass
        idx = 0

        while idx < args.nos + skip_lim:
            world.tick()
            try:
                vehicle.set_autopilot(True)
                ms, kmh = su.speed_estimation(vehicle)

                # Set global speed difference
                if kmh > 24:
                    traffic_manager.global_percentage_speed_difference(sd)
                else:
                    traffic_manager.global_percentage_speed_difference(0.0)

                for _ in range(len(sensor_list)):
                    s_frame = sensor_queue.get(True, 1.0)
                    if s_frame[1] == 1:
                        if idx >= skip_lim:
                            su.record_data(s_frame[0], world, vehicle, features, img_dir)
                        else:
                            pass
                    else:
                        if args.cv_stream:
                            su.live_cam(s_frame[0], dic.cam_spectate_1, pp1=False)
                        else:
                            pass

                idx += 1

            # Stopping key
            except KeyboardInterrupt:
                print("Terminated")
                break

            except Empty:
                pass

    finally:
        print("Finally...")

        # Switch back to synchronous mode
        settings.synchronous_mode = False
        world.apply_settings(settings)
        try:
            # export collected data to csv file
            to.export_csv(csv_file, features)

            # Destroying actors
            print("Destroying {} actor(s)".format(len(actor_list)))
            client.apply_batch([carla.command.DestroyActor(x) for x in sensor_list])
            client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])

        except Exception as e:
            print("Final Exception: ", e)


def main():
    parser = argparse.ArgumentParser(description='Demonstration setup')
    parser.add_argument('-r', help='reload world',     dest='reload',    type=s2b, default='false')
    parser.add_argument('-o', help='opencv stream',    dest='cv_stream', type=s2b, default='false')
    parser.add_argument('-s', help='number of sample', dest='nos',       type=int, default=1000)
    args = parser.parse_args()

    game_loop(args)
    time.sleep(0.5)
    to.overall_check(img_dir, csv_file)


if __name__ == '__main__':
    main()
