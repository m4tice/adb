import blockweek_ad.ca_utils.training_util as tu
import blockweek_ad.ca_utils.tools as to
import os
import random
import numpy as np


img_dir = "../stream_a/collected_data/data3"
csv_file = os.path.join(img_dir, "driving_log.csv")
# new_csv = os.path.join(img_dir, "driving_log2.csv")

to.playback(img_dir, csv_file, pt=0.0001, pp1=True)

to.overall_check("../stream_a/collected_data/data3/IMG", csv_file)
