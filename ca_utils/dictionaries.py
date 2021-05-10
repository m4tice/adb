# variables
img_width = 320
img_height = 160
public_sensor_tick = 0.05


# Colors
orange = '245, 85, 1'
petronas_color = '1, 128, 118'
redbull_color = '18, 20, 45'


# Maps
town01 = {
    'name': 'Town01',
    'dir_name': 'town01',
    'x': 200,
    'y': 157,
    'z': 330,
    'pitch': -90,
    'yaw': 90,
    'weather': 'CloudyNoon'
}

town02 = {
    'name': 'Town02',
    'dir_name': 'town02',
    'x': 90,
    'y': 200,
    'z': 220,
    'pitch': -90,
    'yaw': 0,
    'weather': 'CloudyNoon'
}

town03 = {
    'name': 'Town03',
    'dir_name': 'town03',
    'x': 35,
    'y': 0,
    'z': 400,
    'pitch': -90,
    'yaw': 180,
    'weather': 'CloudyNoon'
}

town04 = {
    'name': 'Town04',
    'dir_name': 'town04',
    'x': -580,
    'y': 200,
    'z': 180,
    'pitch': -45,
    'yaw': 0,
    'weather': 'ClearNoon'
}

town05 = {
    'name': 'Town05',
    'dir_name': 'town05',
    'x': 0,
    'y': -10,
    'z': 420,
    'pitch': -90,
    'yaw': 90,
    'weather': 'CloudyNoon'
}

town06 = {
    'name': 'Town06',
    'dir_name': 'town06',
    'x': 150,
    'y': 120,
    'z': 550,
    'pitch': -90,
    'yaw': 90,
    'weather': 'CloudyNoon'
}

town07 = {
    'name': 'Town07',
    'dir_name': 'town07',
    'x': -70,
    'y': -20,
    'z': 300,
    'pitch': -90,
    'yaw': 0,
    'weather': 'CloudyNoon'
}

town10 = {
    'name': 'Town10HD',
    'dir_name': 'town10',
    'x': -20,
    'y': 30,
    'z': 220,
    'pitch': -90,
    'yaw': 90,
    'weather': 'CloudyNoon'
}

# Courses
course1 = {
    'name': "course1",
    'filename': "course1.csv",
    'init_yaw': 180
}

course2 = {
    'name': "course2",
    'filename': "course2.csv",
    'init_yaw': 180
}

course3 = {
    'name': "course3",
    'filename': "course3.csv",
    'init_yaw': 180
}

course4 = {
    'name': "course4",
    'filename': "course4.csv",
    'init_yaw': 180
}

course5 = {
    'name': "course5",
    'filename': "course5.csv",
    'init_yaw': 128
}

course6 = {
    'name': "course6",
    'filename': "course6.csv",
    'init_yaw': 128
}

course7 = {
    'name': "course7",
    'filename': "course7.csv",
    'init_yaw': 128
}

course8 = {
    'name': "course8",
    'filename': "course8.csv",
    'init_yaw': 128
}

# Weathers
weather = {
    'weathers': ['Default', 'ClearNoon', 'CloudyNoon', 'WetNoon', 'WetCloudyNoon',
                 'MidRainyNoon', 'HardRainNoon', 'SoftRainNoon', 'ClearSunset', 'CloudySunset',
                 'WetSunset', 'WetCloudySunset', 'MidRainSunset', 'HardRainSunset', 'SoftRainSunset']
}

# Sensors
cam_rgb_set_1 = {
    'name': 'sensor.camera.rgb',
    'image_size_x': str(img_width),
    'image_size_y': str(img_height),
    'fov': '110',
    'sensor_tick': str(public_sensor_tick),
    'loc_x': 2.3,
    'loc_y': 0.0,
    'loc_z': 0.8,
    'rot_pitch': 0.0,
    'rot_yaw': 0.0,
    'rot_roll': 0.0
}

cam_rgb_set_2 = {
    'name': 'sensor.camera.rgb',
    'image_size_x': str(img_width),
    'image_size_y': str(img_height),
    'fov': '110',
    'sensor_tick': str(public_sensor_tick),
    'loc_x': 2.3,
    'loc_y': 0.0,
    'loc_z': 0.8,
    'rot_pitch': -10.0,
    'rot_yaw': 0.0,
    'rot_roll': 0.0
}

cam_rgb_set_3 = {
    'name': 'sensor.camera.rgb',
    'image_size_x': str(img_width),
    'image_size_y': str(img_height),
    'fov': '110',
    'sensor_tick': str(public_sensor_tick),
    'loc_x': 2.0,
    'loc_y': 0.0,
    'loc_z': 1.0,
    'rot_pitch': 0.0,
    'rot_yaw': 0.0,
    'rot_roll': 0.0
}

cam_spectate_1 = {
    'name': 'sensor.camera.rgb',
    'image_size_x': str(320),
    'image_size_y': str(160),
    'fov': '110',
    'sensor_tick': str(public_sensor_tick),
    'loc_x': -7.0,  # 1.2 - 2.0 - 2.5
    'loc_y': 0.0,
    'loc_z': 5.0,  # 1.7 - 1.2 - 0.7
    'rot_pitch': -10.0,  # -10.0 - -20.0 - 0.0
    'rot_yaw': 0.0,
    'rot_roll': 0.0
}

cam_spectate_2 = {
    'name': 'sensor.camera.rgb',
    'image_size_x': str(480),
    'image_size_y': str(240),
    'fov': '110',
    'sensor_tick': str(public_sensor_tick),
    'loc_x': -5.5,  # 1.2 - 2.0 - 2.5
    'loc_y': 0.0,
    'loc_z': 2.8,  # 1.7 - 1.2 - 0.7
    'rot_pitch': -15.0,  # -10.0 - -20.0 - 0.0
    'rot_yaw': 0.0,
    'rot_roll': 0.0
}
