
from Environment.search_env import *
from Training.training_helper import *
import tensorflow as tf
from A2C import *
from ddrqn3 import *

"""
Trains the searching network to navigate to target

For best results: train first on local target (converges in 150 episodes), then update search_episode() to train 
on region target (converges in 500 episodes)
                  
Saves plotting to Training_results/Search
"""


def train_search_agent(weights=None):
    # Initialize environment and ddrqn agent
    search = Search()
    action_size = search.num_actions

    sess = tf.Session()
    #searching_agent = A2CAgent(search.vision_size + 6, action_size, 'Search', sess)
    searching_agent = DDRQNAgent(search.vision_size + 6, action_size, 'Search', sess)
    sess.run(tf.global_variables_initializer())
    if weights is not None:
        searching_agent.load(weights+'_model', weights+'_target')



    done = False
    batch_size = 32

    # Initialize episode logging
    episode_rewards = []
    episode_covered = []
    episode_steps = []
    average_over = int(config.num_episodes/10)
    average_rewards = []
    average_r = deque(maxlen=average_over)

    for episode in range(config.num_episodes):
        search.reset_env()
        reward, steps, row_position, col_position = search_episode(search, searching_agent, batch_size,
                                                                   search.row_position, search.col_position)
        episode_rewards.append(reward)
        episode_covered.append(search.calculate_covered('mining'))
        episode_steps.append(steps)

        average_r.append(reward)

        if episode < average_over:
            r = 0
            for i in range(episode):
                r += average_r[i]
            r /= (episode + 1)
            average_rewards.append(r)
        else:
            average_rewards.append(sum(average_r) / average_over)

        if episode % average_over == 0:
            save_plots(episode+1, search, 'Search_test', average_rewards, episode_rewards, mining_coverage=episode_covered)

        print("episode: {}/{}, reward: {}, percent covered: {}, start position: {},{}, number of steps: {}"
              .format(episode+1, config.num_episodes, reward, episode_covered[episode], search.start_row,
                      search.start_col, steps))

        if episode % 30 == 0 and episode != 0:
            searching_agent.save('full_search_weights_model', 'full_search_weights_target', episode)
