import numpy as np
import matplotlib.pyplot as plt
# from dqn_pytorch import Agent
from DDPG.ddpg_torch import Agent
from disassembly_env_v0 import DisassemblyEnv
from Plot_data_diassembly_v0 import *
from save_result_as_csv import *
import pybullet as p


def run_episode(env, agent, max_steps_per_episode):
    state = env.reset()
    total_reward = 0
    step_count = 0
    done = False

    while not done and step_count < max_steps_per_episode:
        action = agent.choose_action(np.array(state))
        next_state, reward, _, done, _ = env.step(action)

        # Store transition in n-step buffer
        agent.remember(state, action, reward, next_state, done)

        if step_count % agent.n_step == 0:
            agent.learn()

        state = next_state
        total_reward += reward
        step_count += 1

    agent.learn()

    return total_reward


if __name__ == '__main__':
    env = DisassemblyEnv()
    agent = Agent(alpha=0.0001, beta=0.001,
                  input_dims=env.observation_space.shape, tau=0.001,
                  batch_size=64, fc1_dims=400, fc2_dims=300,
                  n_actions=env.action_space.shape[0], n_step=5)  # n_step set here

    episode_rewards = []
    figure_file = 'plots/reward.png'
    episodes = 2000

    max_steps_per_episode = 40

    for episode in range(episodes):
        reward = run_episode(env, agent, max_steps_per_episode)
        episode_rewards.append(reward)
        print(f"Episode {episode + 1}: Total Reward = {reward}")

    plt.plot(episode_rewards)
    plt.title('Episode Rewards Over Time')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    # plt.show()
    plt.savefig(figure_file)
