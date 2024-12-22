#!/usr/bin/env python3
'''
    By Xiazhen Xu <xxx212@student.bham.ac.uk>
'''

import csv
import numpy as np
import os

script_dir = os.path.dirname(__file__)
results_dir = os.path.join(script_dir, "results/")
sample_file_name = "Disassembly_results"
sample_file_name_2 = "Force_realtime"


def build_reward_episode(episode, reward):
    data1 = episode
    data2 = reward

    with open('result_csv/railway_rl_reward.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)

        # write the header
        # writer.writerow(header)

        for w in range(len(data1)):
            # print(w)
            writer.writerow([data1[w], data2[w]])


def build_action_step(step, action):
    data1 = step
    data2 = action

    with open('result_csv/railway_rl_action.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)

        # write the header
        # writer.writerow(header)

        for w in range(len(data1)):
            # print(w)
            writer.writerow([data1[w], data2[w]])