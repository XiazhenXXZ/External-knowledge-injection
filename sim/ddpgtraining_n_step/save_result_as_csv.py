import csv
import numpy as np


def build_csv(episode, avg_body_angle, avg_motor1, avg_motor2, avg_reward, avg_reward_):
    # header = ['episode', 'body_angle', 'motor1', 'motor2', 'reward']
    # length_ep = episode_num
    data1 = episode
    data2 = avg_body_angle
    # print(data2)
    data3 = avg_motor1
    data4 = avg_motor2
    data5 = avg_reward
    data6 = avg_reward_
    # print(data5)

    with open('RG_result/curlup.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)

        # write the header
        # writer.writerow(header)

        for w in range(len(data1)):
            # print(w)
            writer.writerow([data1[w], data2[w], data3[w], data4[w], data5[w], data6[w]])


def body_angle_csv(step, body_angle_of_the_best_episode, motor1_of_the_best_episode,
                   motor2_of_the_best_episode):
    # header = ['episode', 'body_angle', 'motor1', 'motor2', 'reward']
    # length_ep = episode_num
    data1 = step
    data2 = body_angle_of_the_best_episode
    # print(data2)
    data3 = motor1_of_the_best_episode
    data4 = motor2_of_the_best_episode

    # print(data5)

    with open('RG_result/dynamics_data.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)

        # write the header
        # writer.writerow(header)

        for w in range(len(data2)):
            # print(w)
            writer.writerow([data1[w], data2[w], data3[w], data4[w]])


def important_value_csv(step, com_y, velocity, com_x):
    # header = ['episode', 'body_angle', 'motor1', 'motor2', 'reward']
    # length_ep = episode_num
    data1 = step
    data2 = com_y
    # print(data2)
    data3 = velocity
    data4 = com_x

    # print(data5)

    with open('RG_result/important_data.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)

        # write the header
        # writer.writerow(header)

        for w in range(len(data2)):
            # print(w)
            writer.writerow([data1[w], data2[w], data3[w], data4[w]])


def all_angle_csv(step, body_angle, motor1,
                  motor2):
    # header = ['episode', 'body_angle', 'motor1', 'motor2', 'reward']
    # length_ep = episode_num
    data1 = step
    data2 = body_angle
    # print(data2)
    data3 = motor1
    data4 = motor2

    # print(data5)

    with open('RG_result/all_angle_data.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)

        # write the header
        # writer.writerow(header)

        for w in range(len(data2)):
            # print(w)
            writer.writerow([data1[w], data2[w], data3[w], data4[w]])
