from collections import defaultdict
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle

from Sailing_Env_v0 import SailingWorld
from Polar_parser import parse

class SailingAgent:
    def __init__(self, env: SailingWorld, learning_rate=0.5, discount_factor=0.9, initial_epsilon=1.0,   
                 final_epsilon=0.01, epsilon_decay=0.9):
        """
        Purpose:
        Initializes a Q-learning agent for the SailingWorld environment.
        It sets up the learning and exploration parameters, and statistic for training.

        Parameters:
            •	env: instance of the SailingWorld environment
            •	learning_rate, discount_factor: standart Q-learning parameters
            •	initial_epsilon: starting value for exploration rate
            •	final_epsilon: minimum exploration rate
            •	epsilon_decay: multiplicative decay rate applied after each episode

        Side effects:
            •	Initializes the Q-table with all zero
            •	Prepares tracking lists for error, rewards, and steps per episode"""
        
        self.env = env

        self.q_values = defaultdict(lambda: {a: 0.0 for a in self.env.action_space})

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.epsilon_decay = epsilon_decay
        self.episode_count = 0

        self.training_error_episode = []
        self.training_error = []
        self.rewards_this_episode = 0
        self.cummulative_rewards_per_episode = []
        self.number_of_steps = 0
        self.steps_per_episode = []


    def get_action(self, state: tuple[float, int], testing=False) -> int:
        """
        Purpose:
        Selects an action for the current state using an epsilon-greedy strategy.

        Parameters:
            •	state: tuple representing the current environment state (rounded x-position, wind angle)
            •	testing (bool): if True, always picks the best known action (no exploration)

        Returns:
            •	int: the chosen action (one of the allowed compass directions)"""

        if testing:
            return max(self.q_values[state], key=self.q_values[state].get) 
        
        if np.random.rand() < self.epsilon:
            return random.choice(self.env.action_space)
        else:
            return max(self.q_values[state], key=self.q_values[state].get)
        
        
    def update(self, state: tuple[float, int], action: int, reward: float, done: bool, next_state: tuple[float, int]):
        """
        Purpose:
            Performs a single Q-learning update step for the observed transition (state, action, reward, next_state).

        Parameters:
            •	state: current state before the action
            •	action: action taken in that state
            •	reward: reward received after taking the action
            •	done: whether the episode has ended (terminal state)
            •	next_state: state resulting from the action

        Behavior:
            •	Computes the TD target
            •	Updates the Q-value
            •	Appends TD error to tracking list
            •	Does statistic things"""
        # TD Target set to zero of done
        target = reward + self.discount_factor * np.max(list(self.q_values[next_state].values())) * (not done)

        # Q-Value Update as discussed in my thesis
        self.q_values[state][action] = self.q_values[state][action] + self.learning_rate * (target - self.q_values[state][action])

        # Statistic things
        self.training_error_episode.append(target - self.q_values[state][action])
        self.number_of_steps += 1
        self.rewards_this_episode += reward


    def update_epsilon(self):
        # reduce epsilon if not minimal
        self.epsilon = max(self.final_epsilon, self.epsilon * self.epsilon_decay)


def train_agent_on(env: SailingWorld, learning_rate = 0.1, discount_factor = 0.99, initial_epsilon = 1.0, final_epsilon = 0.001, 
             n_episodes_max = 10_000, epsilon_decay = 0.999):
    """
    Purpose:
        Trains a SailingAgent on a given SailingWorld environment using Q-learning over multiple episodes.

    Parameters:
        •	env: instance of SailingWorld
        •	learning_rate, discount_factor: Q-learning parameters
        •	initial_epsilon, final_epsilon, epsilon_decay: exploration parameters
        •	n_episodes_max: how many episodes to train

    Behavior:
        1.	Initializes the agent.
        2.	Runs n_episodes_max episodes:
            •	Resets env and agent stats.
            •	Runs steps until episode ends (terminated or truncated).
        3.	After each episode:
            •	Saves average TD error, total reward, and steps.
            •	Decays epsilon.
        4.	After training:
            •	Shows plots of training error, rewards, and steps.
            •	Saves Q-table as q_matrix.pkl.
    """
    env.reset()

    # Initializing the agent
    agent = SailingAgent(env=env, learning_rate=learning_rate, discount_factor=discount_factor, 
                        initial_epsilon=initial_epsilon, final_epsilon=final_epsilon, epsilon_decay=epsilon_decay)


    # Using tqdm to loop over episodes for progress bar
    for n_episode in tqdm(range(n_episodes_max)):

        # Statistic Things
        agent.training_error_episode = []
        agent.number_of_steps = 0
        agent.rewards_this_episode = 0

        state = env.reset()
        done = False

        # Computing an Episode
        while not done:
            action = agent.get_action(state=state)
            new_state, terminated, reward, truncated = env.step(action)
            done = terminated or truncated
            agent.update(state=state, action=action, next_state=new_state, done=done, reward=reward)
            state = new_state

        # Taking the mean over all TD Errors of an episode (statistics)
        agent.training_error.append(np.mean(agent.training_error_episode))
        agent.steps_per_episode.append(agent.number_of_steps)
        agent.cummulative_rewards_per_episode.append(agent.rewards_this_episode)
        
        agent.update_epsilon()

    print("Training done")

    # Displaying the collected data in dependece on the episode count
    x = list(range(1, n_episodes_max + 1))

    fig, axs = plt.subplots(3, 1, figsize=(16, 12))
    fig.tight_layout(pad=8.0)

    axs[0].scatter(x, agent.training_error, s=3)
    axs[0].set_xlabel('Episode')
    axs[0].set_ylabel('Average Target Error')
    axs[0].set_title('Average Target Error per Episode')

    axs[1].scatter(x, agent.cummulative_rewards_per_episode, s=3)
    axs[1].set_xlabel('Episode')
    axs[1].set_ylabel('Average Reward')
    axs[1].set_title('Average Reward per Episode')

    axs[2].scatter(x, agent.steps_per_episode, s=3)
    axs[2].set_xlabel('Episode')
    axs[2].set_ylabel('Number of Steps')
    axs[2].set_title('Number of Steps per Episode')
    plt.show()

    print("Done with boring statistcs")

    # Saving the Q-Matrix
    with open("q_matrix.pkl", "wb") as f:
        pickle.dump(dict(agent.q_values), f)  # defaultdict -> dict
    print("Q-Matrix gespeichert als q_matrix.pkl")
    
    return agent


def testing(agent: SailingAgent, env: SailingWorld):
    """
    Purpose:
        Runs the trained agent in test mode (no exploration) to visualize its learned behavior.

    Behavior:
        •	Resets environment and agent reward counter.
        •	Repeatedly:
            •	Executes the best known action.
            •	Updates cumulative reward and prints it.
            •	Renders the environment.

    Notes:
        •	No learning is performed.
        •	This is just for evaluation and visual inspection."""
    
    print("Testing starts now...")
    state = env.reset()
    done = False
    agent.rewards_this_episode = 0
    while not done:
        action = agent.get_action(state=state, testing=True)
        new_state, terminated, reward, truncated = env.step(action)
        print("reward:", reward)
        agent.rewards_this_episode += reward
        print("com. reward:", agent.rewards_this_episode)
        done = terminated or truncated 
        env.render_human(agent.rewards_this_episode)  
        state = new_state
    env.render_human()

if __name__ == "__main__":
    print("Welcome to the Sailing environment!")
    Env = SailingWorld(width=10, length=15, delta_t=10, fps=60)
    if input("What do you want to do? (train/test): ").lower() == 'train':
        Agent = train_agent_on(Env, n_episodes_max=500_000, epsilon_decay=0., final_epsilon=0.1, learning_rate=0.1, initial_epsilon=1.0, discount_factor=0.99)
    else:
        # Load the Q-Matrix saved on train_agent_on(...)
        with open("q_matrix.pkl", "rb") as f:
            q_values = pickle.load(f)
        Agent = SailingAgent(env=Env)
        Agent.q_values = defaultdict(lambda: {a: 0.0 for a in Env.action_space}, q_values)
        while input("Do you want to start a test? (y/n): ").lower() == 'y':
            testing(Agent, Env)

    Env.close()