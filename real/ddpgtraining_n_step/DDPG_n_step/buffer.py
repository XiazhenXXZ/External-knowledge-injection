import numpy as np
from collections import deque, namedtuple


class NStepReplayBuffer:
    def __init__(self, max_size, input_dims, n_actions, n_step=5, gamma=0.99):
        self.memory = deque(maxlen=max_size)
        self.n_step = n_step
        self.gamma = gamma
        self.n_step_buffer = deque(maxlen=n_step)
        self.input_dims = input_dims
        self.n_actions = n_actions

    def store_transition(self, state, action, reward, state_, done):
        transition = (state, action, reward, state_, done)
        self.n_step_buffer.append(transition)

        if len(self.n_step_buffer) == self.n_step:
            cumulative_reward, n_state_, n_done = self._get_n_step_info()
            first_state, first_action = self.n_step_buffer[0][:2]
            # Store the first transition with n-step cumulative reward and next state
            self.memory.append((first_state, first_action, cumulative_reward, n_state_, n_done))

    def _get_n_step_info(self):
        cumulative_reward = 0
        for idx, (_, _, reward, _, done) in enumerate(self.n_step_buffer):
            cumulative_reward += (self.gamma ** idx) * reward
            if done:
                break
        n_state_, n_done = self.n_step_buffer[-1][3], self.n_step_buffer[-1][4]
        return cumulative_reward, n_state_, n_done

    def sample_buffer(self, batch_size):
        if len(self.memory) < batch_size:
            return None, None, None, None, None

        batch = np.random.choice(len(self.memory), batch_size, replace=False)
        states, actions, rewards, states_, dones = [], [], [], [], []

        for idx in batch:
            state, action, reward, state_, done = self.memory[idx]
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            states_.append(state_)
            dones.append(done)

        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.float32),
            np.array(rewards, dtype=np.float32),
            np.array(states_, dtype=np.float32),
            np.array(dones, dtype=bool)
        )

    def __len__(self):
        return len(self.memory)


