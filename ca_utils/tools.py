import matplotlib.pyplot as plt
import pandas as pd

import os
import cv2
import csv

import blockweek_ad.ca_utils.training_util as tu


def load_course_data(csv_file):
    """
    Load course (town04 only)
    :param csv_file:
    :return:
    """
    data_df = pd.read_csv(csv_file, names=['x', 'y'])
    data = data_df[['x', 'y']].values

    return data


def load_camera_data(csv_file):
    """
    Load recorded data
    """
    data_df = pd.read_csv(csv_file,
                          names=['img', 'loc_x', 'loc_y', 'atTrafficLight', 'left_lane', 'right_lane', 'throttles',
                                 'braking', 'steering'])
    data = data_df[['img', 'loc_x', 'loc_y', 'atTrafficLight', 'left_lane', 'right_lane', 'throttles', 'braking',
                    'steering']].values

    return data


def export_csv(csv_file, features):
    """
    Export vehicle data
    """
    print("Calling: exporting vehicle data (writing into csv file!)")

    if os.path.isfile(csv_file):
        with open(csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            for img_name, loc_x, loc_y, atTrafficLight, left_lane, right_lane, throttle, brake, steer in features:
                writer.writerow([img_name, loc_x, loc_y, atTrafficLight, left_lane, right_lane, throttle, brake, steer])
    else:
        with open(csv_file, 'w', newline='') as file:
            writer = csv.writer(file)
            for img_name, loc_x, loc_y, atTrafficLight, left_lane, right_lane, throttle, brake, steer in features:
                writer.writerow([img_name, loc_x, loc_y, atTrafficLight, left_lane, right_lane, throttle, brake, steer])


def playback(img_dir, csv_file, pt=0.001, pp1=False):
    """
    display the recorded images
    """
    data = load_camera_data(csv_file)
    fig, ax = plt.subplots()

    for i, sample in enumerate(data):
        ax.cla()
        ax.set_title(i)
        image_name = sample[0]
        image = tu.load_carla_image(image_name, path=img_dir)

        if pp1:
            image = tu.preprocess1(image)

        # cv2.imshow("playback", image)
        # cv2.waitKey(25)
        ax.imshow(image)
        plt.pause(pt)


# == DATA ERROR CHECKING =====
def redundant_img_check(img_dir, csv_file):
    """
    Checking for redundant sample
    """
    if os.path.isdir(img_dir):
        print("Calling: redundant image check")
        temp = os.listdir(img_dir)
        img_list = [[item, False] for item in temp]

        data = load_camera_data(csv_file)
        temp = data[:, 0]
        data = [item.split("/") for item in temp]
        data = [item[1] for item in data]

        for item in img_list:
            for sample in data:
                if item[0] == sample:
                    item[1] = True
                    break

        rm_list = []
        for item in img_list:
            if not item[1]:
                rm_list.append(item[0])
                os.remove(img_dir + "/" + item[0])

        if len(rm_list) > 0:
            print("- Removed {} item(s)".format(len(rm_list)))
            print("- Removed:", rm_list)

    else:
        print("- Directory not found!")


def existence_check(img_dir, csv_file):
    """
    Checking for non-existent sample
    """
    print("Calling: existent check")
    data = load_camera_data(csv_file)
    temp = data[:, 0]
    data = [item.split("/") for item in temp]
    data = [item[1] for item in data]

    pms = []
    for item in data:
        pm = os.path.isfile(os.path.join(img_dir, item))
        pms.append(pm)

    return all(pms)


def count_line_csv(csv_file):
    """
    Count labels
    """
    with open(csv_file, 'r', newline='') as file:
        file_object = csv.reader(file)
        row_count = sum(1 for row in file_object)  # fileObject is your csv.reader
        return row_count


# -
def images_count(directory):
    """
    Count images
    """
    count = len([img for img in os.listdir(directory)])
    return count


def data_balance_check(dir, csv_file):
    """
    Check for balance between features and labels
    """
    print("Calling: data balance checking")
    img_num = images_count(dir)
    line_num = count_line_csv(csv_file)
    print("-  Count: Features: {} - Labels: {}".format(img_num, line_num))

    if img_num == line_num and img_num != 0 and line_num != 0:
        print('- Result: Balanced data.')
        return True
    else:
        print('- Result: Unbalanced data.')
        return False


def overall_check(img_dir, csv_file):
    """
    Overall check for balance and existence
    """
    print("Calling: Data checking")
    redundant_img_check(img_dir, csv_file)
    print(existence_check(img_dir, csv_file))
    data_balance_check(img_dir, csv_file)


def s2b(s):
    """
    Converts a string to boolean value
    """
    s = s.lower()
    return s == 'true' or s == 'yes' or s == 'y' or s == '1'
