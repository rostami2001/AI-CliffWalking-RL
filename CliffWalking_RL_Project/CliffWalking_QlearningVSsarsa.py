import gymnasium as gym
import numpy as np
import pickle as pkl
import cv2
from PIL import Image
import matplotlib.pyplot as plt

cliffEnv = gym.make("CliffWalking-v0")
reward_cache_qlearning = []
step_cache_qlearning = []
training_error_qlearning = []  # For storing TD errors (2)

q_table = np.zeros(shape=(cliffEnv.observation_space.n, cliffEnv.action_space.n))

# Parameters
EPSILON = 1.0  # (1)
EPSILON_DECAY = 0.995  # (1)
EPSILON_MIN = 0.1  # (1)
ALPHA = 0.1
GAMMA = 0.9
NUM_EPISODES = 500


def policy(state, explore=0.0):
    # exploration vs exploitation
    if np.random.random() <= explore:
        # Explore
        action = int(np.random.randint(low=0, high=cliffEnv.action_space.n, size=1))
    else:
        # Exploit
        action = int(np.argmax(q_table[state]))
    return action


for episode in range(NUM_EPISODES):

    done = False

    total_reward = 0
    episode_length = 0

    state, _ = cliffEnv.reset()

    while not done:
        action = policy(state, EPSILON)
        next_state, reward, done, truncated, _ = cliffEnv.step(action)

        next_action = policy(next_state, EPSILON)

        # Q(next_state, next_action)
        Q_nextSA = np.max(q_table[next_state])
        # Q_nextSA = q_table[next_state][next_action]
        # Temporal Difference
        TDerror = reward + GAMMA * Q_nextSA - q_table[state][action]
        training_error_qlearning.append(TDerror)  # (2)
        # Update Q-table for this state action pair
        q_table[state][action] += ALPHA * TDerror

        state = next_state

        total_reward += reward
        episode_length += 1

    reward_cache_qlearning.append(total_reward)
    step_cache_qlearning.append(episode_length)

    # Decay epsilon
    if EPSILON > EPSILON_MIN:
        EPSILON *= EPSILON_DECAY  # (1)

    print("Episode:", episode, "Episode Length:",
          episode_length, "Total Reward:", total_reward)

cliffEnv.close()

pkl.dump(q_table, open("q_learning_q_table.pkl", "wb"))

print("Training Complete, Q Table Saved :)")

print('Q-table', q_table)

# *** Q-learning Visualization ***
cliffEnv = gym.make("CliffWalking-v0")

q_table = pkl.load(open("q_learning_q_table.pkl", "rb"))


# Creates cliff walking grid
def initialize_frame():
    width, height = 600, 200
    img = np.ones(shape=(height, width, 3)) * 255.0
    margin_horizontal = 6
    margin_vertical = 2

    # Vertical Lines
    for i in range(13):
        img = cv2.line(img, (49 * i + margin_horizontal, margin_vertical),
                       (49 * i + margin_horizontal, 200 - margin_vertical), color=(0, 0, 0), thickness=1)

    # Horizontal Lines
    for i in range(5):
        img = cv2.line(img, (margin_horizontal, 49 * i + margin_vertical),
                       (600 - margin_horizontal, 49 * i + margin_vertical), color=(0, 0, 0), thickness=1)

    # Cliff Box
    img = cv2.rectangle(img, (49 * 1 + margin_horizontal + 2, 49 * 3 + margin_vertical + 2),
                        (49 * 11 + margin_horizontal - 2, 49 * 4 + margin_vertical - 2), color=(255, 0, 255),
                        thickness=-1)
    img = cv2.putText(img, text="Cliff", org=(49 * 5 + margin_horizontal, 49 * 4 + margin_vertical - 10),
                      fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

    # Goal
    frame = cv2.putText(img, text="G", org=(49 * 11 + margin_horizontal + 10, 49 * 4 + margin_vertical - 10),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0), thickness=2)
    return frame


# puts the agent at a state
def put_agent(img, state):
    margin_horizontal = 6
    margin_vertical = 2
    row, column = divmod(state, 12)
    cv2.putText(img, text="A", org=(49 * column + margin_horizontal + 10, 49 * (row + 1) + margin_vertical - 10),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0), thickness=2)
    return img


def policy(state, explore=0.0):
    action = int(np.argmax(q_table[state]))
    if np.random.random() <= explore:
        action = int(np.random.randint(low=0, high=4, size=1))
    return action


NUM_EPISODES = 3

frames = []  # for gif

for episode in range(NUM_EPISODES):

    done = False

    total_reward = 0
    episode_length = 0

    frame = initialize_frame()
    state, _ = cliffEnv.reset()

    while not done:
        frame2 = put_agent(frame.copy(), state)

        # Append the frame as a PIL Image
        frames.append(Image.fromarray(frame2.astype('uint8')))

        cv2.imshow("Cliff Walking", frame2)
        cv2.waitKey(250)
        action = policy(state)
        state, reward, done, _, _ = cliffEnv.step(action)

        total_reward += reward
        episode_length += 1

    print("Episode:", episode, "Episode Length:",
          episode_length, "Reward:", total_reward)

    # Close the display window after each episode
    cv2.destroyAllWindows()

# Save the frames as a GIF
frames[0].save('Q-Learning Agent.gif', format='GIF',
               append_images=frames[1:], save_all=True, duration=250)

cliffEnv.close()

# Load the environment and Q-table
cliffEnv = gym.make("CliffWalking-v0")
q_table = pkl.load(open("q_learning_q_table.pkl", "rb"))


# Define a function to plot the Q-table and actions
def plot_q_table(q_table):
    action_symbols = ['↑', '→', '↓', '←']
    action_arrows = [(0, -1), (1, 0), (0, 1), (-1, 0)]

    # Create a blank grid
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(-0.5, 11.5)
    ax.set_ylim(-0.5, 3.5)

    # Plot the grid lines
    for x in range(12):
        ax.axvline(x - 0.5, color='black', linewidth=1)
    for y in range(4):
        ax.axhline(y - 0.5, color='black', linewidth=1)

    # Plot the cliff
    for x in range(1, 11):
        ax.add_patch(plt.Rectangle((x - 0.5, 3 - 0.5), 1, 1, color='purple'))
        ax.text(x, 3, 'Cliff', ha='center', va='center', color='white', fontsize=10)

    # Plot the start and goal states
    ax.text(0, 3, 'S', ha='center', va='center', color='black', fontsize=15, fontweight='bold')
    ax.text(11, 3, 'G', ha='center', va='center', color='black', fontsize=15, fontweight='bold')

    # Plot the actions for each state
    for state in range(q_table.shape[0]):
        row, col = divmod(state, 12)
        if (row, col) == (3, 0) or (row, col) == (3, 11):  # Skip start and goal states
            continue
        best_action = np.argmax(q_table[state])
        ax.text(col, 3 - row, action_symbols[best_action], ha='center', va='center', color='black', fontsize=15)

    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()


# Call the function to plot the Q-table
plot_q_table(q_table)

# *** SARSA ***
cliffEnv = gym.make("CliffWalking-v0")

q_table = np.zeros(shape=(cliffEnv.observation_space.n, cliffEnv.action_space.n))
reward_cache_SARSA = []
step_cache_SARSA = []
training_error_SARSA = []  # For storing TD errors (2)

# Parameters
EPSILON = 1.0  # (1)
EPSILON_DECAY = 0.995  # (1)
EPSILON_MIN = 0.1  # (1)
ALPHA = 0.1
GAMMA = 0.9

NUM_EPISODES = 500


def policy(state, explore=0.0):
    if np.random.random() <= explore:
        # Explore
        action = int(np.random.randint(low=cliffEnv.action_space.n))
    else:
        # Exploit
        action = int(np.argmax(q_table[state]))
    return action



for episode in range(NUM_EPISODES):

    done = False
    total_reward = 0
    episode_length = 0

    state, _ = cliffEnv.reset()
    action = policy(state, EPSILON)

    while not done:
        next_state, reward, done, truncated, _ = cliffEnv.step(action)

        next_action = policy(next_state, EPSILON)

        # Update Q-table
        TDerror = reward + GAMMA * q_table[next_state][next_action] - q_table[state][action]
        training_error_SARSA.append(TDerror)  # (2)
        q_table[state][action] += ALPHA * TDerror

        state = next_state
        action = next_action

        total_reward += reward
        episode_length += 1

    reward_cache_SARSA.append(total_reward)
    step_cache_SARSA.append(episode_length)

    # Decay epsilon
    if EPSILON > EPSILON_MIN:
        EPSILON *= EPSILON_DECAY  # (1)

    print("Episode:", episode, "Episode Length:", episode_length, "Total Reward:", total_reward)

cliffEnv.close()

pkl.dump(q_table, open("sarsa_q_table.pkl", "wb"))

print("Training Complete, Q Table Saved :)")
print("Q-table", q_table)

# SARSA Visualization
cliffEnv = gym.make("CliffWalking-v0")

q_table = pkl.load(open("sarsa_q_table.pkl", "rb"))

NUM_EPISODES = 3

frames = []

for episode in range(NUM_EPISODES):

    done = False
    total_reward = 0
    episode_length = 0

    frame = initialize_frame()
    state, _ = cliffEnv.reset()

    while not done:
        frame2 = put_agent(frame.copy(), state)

        # Append the frame as a PIL Image
        frames.append(Image.fromarray(frame2.astype('uint8')))

        cv2.imshow("Cliff Walking", frame2)
        cv2.waitKey(250)
        action = policy(state)
        state, reward, done, truncated, _ = cliffEnv.step(action)

        total_reward += reward
        episode_length += 1

    print("Episode:", episode, "Episode Length:", episode_length, "Reward:", total_reward)

    # Close the display window after each episode
    cv2.destroyAllWindows()

# Save the frames as a GIF
frames[0].save('SARSA Agent.gif', format='GIF',
               append_images=frames[1:], save_all=True, duration=250)

cliffEnv.close()


# *** Plot Performance Q-learning VS SARSA
def plot_cumreward_normalized(reward_cache_qlearning, reward_cache_SARSA):
    """
    Visualizes the reward convergence

    Args:
        reward_cache -- type(list) contains cumulative_reward
    """
    cum_rewards_q = []
    rewards_mean_q = np.array(reward_cache_qlearning).mean()
    rewards_std_q = np.array(reward_cache_qlearning).std()
    count = 0  # used to determine the batches
    cur_reward = 0  # accumulate reward for the batch
    for cache in reward_cache_qlearning:
        count = count + 1
        cur_reward += cache
        if (count == 10):
            # normalize the sample
            normalized_reward = (cur_reward - rewards_mean_q) / rewards_std_q
            cum_rewards_q.append(normalized_reward)
            cur_reward = 0
            count = 0

    cum_rewards_SARSA = []
    rewards_mean_sarsa = np.array(reward_cache_SARSA).mean()
    rewards_std_sarsa = np.array(reward_cache_SARSA).std()
    count = 0  # used to determine the batches
    cur_reward = 0  # accumulate reward for the batch
    for cache in reward_cache_SARSA:
        count = count + 1
        cur_reward += cache
        if (count == 10):
            # normalize the sample
            normalized_reward = (cur_reward - rewards_mean_sarsa) / rewards_std_sarsa
            cum_rewards_SARSA.append(normalized_reward)
            cur_reward = 0
            count = 0
            # prepare the graph
    plt.plot(cum_rewards_q, label="Q-learning")
    plt.plot(cum_rewards_SARSA, label="SARSA")
    plt.ylabel('Cumulative Rewards')
    plt.xlabel('Batches of Episodes (sample size 10) ')
    plt.title("Q-Learning/SARSA Convergence of Cumulative Reward")
    plt.legend(loc='lower right', ncol=2, mode="expand", borderaxespad=0.)
    plt.show()


def plot_number_steps(step_cache_qlearning, step_cache_SARSA):
    """
        Visualize number of steps taken
    """
    cum_step_q = []
    steps_mean_q = np.array(step_cache_qlearning).mean()
    steps_std_q = np.array(step_cache_qlearning).std()
    count = 0  # used to determine the batches
    cur_step = 0  # accumulate reward for the batch
    for cache in step_cache_qlearning:
        count = count + 1
        cur_step += cache
        if (count == 10):
            # normalize the sample
            normalized_step = (cur_step - steps_mean_q) / steps_std_q
            cum_step_q.append(normalized_step)
            cur_step = 0
            count = 0

    cum_step_SARSA = []
    steps_mean_sarsa = np.array(step_cache_SARSA).mean()
    steps_std_sarsa = np.array(step_cache_SARSA).std()
    count = 0  # used to determine the batches
    cur_step = 0  # accumulate reward for the batch
    for cache in step_cache_SARSA:
        count = count + 1
        cur_step += cache
        if (count == 10):
            # normalize the sample
            normalized_step = (cur_step - steps_mean_sarsa) / steps_std_sarsa
            cum_step_SARSA.append(normalized_step)
            cur_step = 0
            count = 0
            # prepare the graph
    plt.plot(cum_step_q, label="Q-learning")
    plt.plot(cum_step_SARSA, label="SARSA")
    plt.ylabel('Number of iterations')
    plt.xlabel('Batches of Episodes (sample size 10) ')
    plt.title("Q-Learning/SARSA Iteration number until game ends")
    plt.legend(loc='lower right', ncol=2, mode="expand", borderaxespad=0.)
    plt.show()


# Plot cumulative rewards
plot_cumreward_normalized(reward_cache_qlearning, reward_cache_SARSA)

# Plot number of steps
plot_number_steps(step_cache_qlearning, step_cache_SARSA)


# *** Comparing Q-learning and SARSA ***
plt.plot(reward_cache_qlearning, label='Q-learning')
plt.plot(reward_cache_SARSA, label='SARSA')
plt.title('Reward Over Episodes')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.legend(loc='lower right')
plt.show()

plt.plot(step_cache_qlearning, label='Q-learning')
plt.plot(step_cache_SARSA, label='SARSA')
plt.title('Steps Per Episode')
plt.xlabel('Episode')
plt.ylabel('Number of Steps')
plt.legend(loc='upper right')
plt.show()

# *** Plotting TD Error ***

plt.plot(training_error_qlearning, label='Q-learning TD Error')
plt.plot(training_error_SARSA, label='SARSA TD Error')
plt.title('Temporal Difference Error Over Time')
plt.xlabel('Time Step')
plt.ylabel('TD Error')
plt.legend(loc='upper right')
plt.show()