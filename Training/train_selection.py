from Environment.target_mock_env import *
from Training.training_helper import *
import tensorflow as tf
from A2C import *
from ddrqn2 import *
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import os


def train_selection():
    target = SelectTargetTrain()

    #sess = tf.Session()

    selection_agent = DDRQNAgent(config.num_targets * 3+1, config.num_targets)

    #sess.run(tf.global_variables_initializer())

    done = False

    #outF = open("Training_results/MockTarget/State_actions.txt", "a")
    batch_size = 32

    average_over = int(config.num_episodes / 10)
    target_selection_rewards = []
    target_selection_average_rewards = []
    target_selection_average_r = deque(maxlen=average_over)
    #states_actions = np.zeros(shape=[config.num_episodes, 50, 2])

    for e in range(config.num_episodes):
        target.reset_env()
        target_selection_reward = 0
        target_selection_state = np.zeros([1, 28])
        episode = []

        Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

        for t in range(10):
            #Choose next target
            states = np.zeros([1, 5, 28])
            local_maps = np.zeros([1, 5, 625])

            if t < 5:
                action = target.select_next_target(target.row_position, target.col_position)
            else:
                states, local_maps = get_last_t_states_target(5, episode, config.num_targets*3+1)
                #action = selection_agent.act(states)
                action = selection_agent.act(target_selection_state)
                #Testing to see reward with cost function
                #action = target.select_next_target(target.row_position, target.col_position)

            #states_actions[e, t, 0] = target.current_target_index
            #states_actions[e, t, 1] = action

            #print(target.current_target_index, action)

            next_target, next_state, reward = target.set_target(action)
            target_selection_reward += reward

            episode.append(Transition(state=target_selection_state, action=action, reward=reward, next_state=next_state, done=done))
            target_selection_state = next_state

            if t > 5:
                next_states, next_local_maps = get_last_t_states_target(5, episode, config.num_targets*3+1)
                #selection_agent.memorize(states, local_maps, action, reward, next_states, next_local_maps, done)
                selection_agent.memorize(target_selection_state, local_maps, action, reward, next_state, next_local_maps, done)
                #value_next = selection_agent.value.predict(next_states, next_local_maps)
                #td_target = reward + .95 * value_next
                #td_error = td_target - selection_agent.value.predict(states, local_maps)
                #selection_agent.update(states, local_maps, td_error, td_target, action)



        target_selection_rewards.append(target_selection_reward)
        target_selection_average_r.append(target_selection_reward)

        if e % 5 == 0 and e != 0:
            selection_agent.update_target_model()

        if len(selection_agent.memory) > batch_size:
            selection_agent.replay(batch_size)


        print("Episode: {} Reward: {}".format(e, target_selection_reward))


        if e < average_over:
            r = 0
            for i in range(e):
                r += target_selection_average_r[i]
            r /= (e + 1)
            target_selection_average_rewards.append(r)
        else:
            target_selection_average_rewards.append(sum(target_selection_average_r) / average_over)

        if e % average_over == 0:
            save_plots(e + 1, target, 'MockTarget', target_selection_average_rewards, target_selection_rewards)

            #NEED A WAY TO SIMULATE LOCAL MAPS - FIGURE OUT WHAT LOCAL MAPS ARE

            #Update rewards

        """
        if e % average_over == 0 and e!= 0:
            x = np.reshape(states_actions, [-1, 100])
            outF.write("-----------------------------")
            np.savetxt("Training_results/MockTarget/State_actions.txt", x[-average_over:, :])
        """

