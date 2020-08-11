from ddrqn3 import *
from ddqn_modified import *
from A2C import *
from Environment.search_env_exp_tar import *
from Environment.tracing_env import *
from Environment.target_selector_env import *
from Testing.testing_helper import *
from Environment.visited import *
from Environment.map import *

"""
Trains the searching network to navigate to target

For best results: train search and trace networks first, then train with target_cost=True (use cost function to 
select next target), then train full network 

Saves plotting to Training_results/Search and Training_results/Trace and Training_results/Target
"""


def train_selection(target_cost=False, search_weights=None, trace_weights=None, target_weights=None):
    # Initialize environment and ddrqn agents
    visited = Visited(config.total_rows, config.total_cols)
    map_obj = Map(config.total_rows, config.total_cols)

    search = Search(visited, map_obj)
    trace = Trace(visited, map_obj)
    target = SelectTarget(visited)

    action_size = search.num_actions
    sess = tf.Session()
    searching_agent = DDRQNAgent(search.vision_size + 6, action_size, 'Search', sess)

    tracing_agent = A2CAgent(trace.vision_size + 4, action_size, 'Trace', sess)

    if not target_cost:
        selection_agent = DDQNAgent(config.num_targets * 3, config.num_targets)

    sess.run(tf.global_variables_initializer())
    if search_weights is not None:
        searching_agent.load(search_weights + '_model', search_weights + '_target')
        print("Search weights loaded")
    if trace_weights is not None:
        tracing_agent.load(trace_weights + '_policy', trace_weights + '_value')
        print("Trace weights loaded")
    if target_weights is not None:
        selection_agent.load(target_weights + '_model', target_weights + '_target')
        print("Target weights loaded")
    done = False
    batch_size = 48

    # Initialize episode logging
    search_rewards = []
    search_covered = []
    search_steps = []
    average_over = 10
    search_average_rewards = []
    search_average_r = deque(maxlen=average_over)
    search_episode_num = 0

    trace_rewards = []
    trace_covered = []
    trace_steps = []
    trace_average_rewards = []
    trace_average_r = deque(maxlen=average_over)
    trace_episode_num = 0


    target_selection_rewards = []
    target_selection_average_rewards = []
    target_selection_average_r = deque(maxlen=average_over)

    num_steps = []
    num_average_steps = []
    num_average_s = deque(maxlen=average_over)

    for e in range(config.num_episodes):
        mining_coverage = []
        search.reset_env(visited, map_obj)
        trace.reset_env(visited, map_obj)
        target.reset_env(visited, map_obj)
        t = 0


        episode = []
        iteration = 0
        Transition = collections.namedtuple("Transition", ["state", "local_map", "action", "reward", "next_state",
                                                           "next_local_map", "done"])
        target_selection_reward = 0
        target_selection_state = np.zeros([1, 27])

        #while iteration < 50:
        while target.calculate_covered('mining') < .7:
            mining = target.calculate_covered('mining')
            print('Mining Coverage:', mining)
            mining_coverage.append(mining)
            print('Total Steps:', t)

            if target_cost:
                next_target = target.select_next_target(trace.row_position, trace.col_position)
                #pass


            else:

                states = np.zeros([1, 5, 27])
                local_maps = np.zeros([1, 5, 625])
                if iteration < 5:
                    action = target.select_next_target(trace.row_position, trace.col_position)
                else:
                    states, local_maps = get_last_t_states(5, episode, config.num_targets * 3)
                    action = selection_agent.act(states, local_maps)

                next_target, next_state, target_reward = target.set_target(action, trace.row_position,
                                                                           trace.col_position)

            # Complete one searching episode
            #print("Start Search: {}".format(np.sum(trace.visited)))
            s_reward, steps, row_position, col_position = search_episode(search, searching_agent,
                                                                       trace.row_position, trace.col_position)
            search_rewards.append(s_reward)
            search_covered.append(search.calculate_covered('mining'))
            search_steps.append(steps)

            search_average_r.append(s_reward)

            if search_episode_num < average_over:
                r = 0
                for i in range(search_episode_num):
                    r += search_average_r[i]
                r /= (search_episode_num + 1)
                search_average_rewards.append(r)
            else:
                search_average_rewards.append(sum(search_average_r) / average_over)

            if search_episode_num % average_over == 0:
                save_plots(e + 1, search, 'Search', search_average_rewards, search_rewards,
                           mining_coverage=search_covered, map_obj=map_obj)

            save_plots(e + 1, search, 'Search', map_obj=map_obj)
            print("search episode: {} - {}, reward: {}, mining covered: {}, start position: {},{}, number of steps: {}"
                  .format((search_episode_num+1) % (e+1), e+1, s_reward, search_covered[search_episode_num],
                          trace.row_position, trace.col_position, steps))

            search_episode_num += 1
            t += steps

            # Update all environments
            #trace.update_visited(search.visited)
            #trace.transfer_map(search.map)
            target.update_visited(search.visited)
            target.transfer_map(search.map)

            # Complete one tracing episode
            t_reward, steps, row_position, col_position = trace_episode(trace, tracing_agent,
                                                                      search.row_position, search.col_position, target)
            trace_rewards.append(t_reward)
            trace_covered.append(trace.calculate_covered('mining'))
            trace_steps.append(steps)

            trace_average_r.append(t_reward)

            if trace_episode_num < average_over:
                r = 0
                for i in range(trace_episode_num):
                    r += trace_average_r[i]
                r /= (trace_episode_num + 1)
                trace_average_rewards.append(r)
            else:
                trace_average_rewards.append(sum(trace_average_r) / average_over)

            if trace_episode_num % average_over == 0:
                save_plots(e + 1, trace, 'Trace', trace_average_rewards, trace_rewards,
                           mining_coverage=trace_covered, map_obj=map_obj)

            save_plots(e + 1, trace, 'Trace', map_obj=map_obj)
            print("trace episode: {} - {}, reward: {}, mining covered: {}, start position: {},{}, number of steps: {}"
                  .format((trace_episode_num+1) % (e+1), e+1, t_reward, trace_covered[trace_episode_num],
                          search.row_position, search.col_position, steps))

            trace_episode_num += 1
            t += steps

            # Update all environments
            #search.update_visited(trace.visited)
            #search.transfer_map(trace.map)
            target.update_visited(trace.visited)
            target.transfer_map(trace.map)

            target_reward = (.8 * s_reward) + t_reward

            if target_cost:
                #next_target = target.select_next_target(trace.row_position, trace.col_position)
                pass


            else:
                """
                states = np.zeros([1, 5, 27])
                local_maps = np.zeros([1, 5, 625])
                if iteration < 5:
                    action = target.select_next_target(trace.row_position, trace.col_position)
                else:
                    states, local_maps = get_last_t_states(5, episode, config.num_targets*3)
                    action = selection_agent.act(states, local_maps)

                next_target, next_state, target_reward = target.set_target(action, trace.row_position, trace.col_position)
                """
                target_selection_reward += target_reward

                next_state = target.get_state()


                episode.append(Transition(
                    state=target_selection_state, local_map=trace.local_map, action=action, reward=target_reward,
                    next_state=next_state, next_local_map=trace.local_map, done=done))

                target_selection_state = next_state

                if iteration > 5:
                    next_states, next_local_maps = get_last_t_states(5, episode, config.num_targets*3)

                    selection_agent.memorize(states, local_maps, action, target_reward, next_states, next_local_maps, done)

                if target.calculate_covered('mining') >= .7:
                    selection_agent.update_target_model()

                if len(selection_agent.memory) > batch_size:
                    selection_agent.replay(batch_size)

                if iteration > 20 and iteration % 20 == 0:
                    selection_agent.decay_learning_rate()

            iteration += 1

            # Update all environments
            search.update_target(next_target)
            trace.update_target(next_target)
            target.update_target(next_target)
            print("Next target:", next_target)


        target_selection_rewards.append(target_selection_reward)
        target_selection_average_r.append(target_selection_reward)

        if e < average_over:
            r = 0
            for i in range(e):
                r += target_selection_average_r[i]
            r /= (e + 1)
            target_selection_average_rewards.append(r)
        else:
            target_selection_average_rewards.append(sum(target_selection_average_r) / average_over)

        if e % average_over == 0:
            save_plots(e + 1, target, 'Target', target_selection_average_rewards, target_selection_rewards, map_obj=map_obj)

        print("trace episode: {}/{}, reward: {}".format(e+1, config.num_episodes, sum(target_selection_rewards)))

        print('***********')
        print("EPISODE {} COMPLETE: Steps -- {}, Mining Coverage -- {}, Total Coverage: {}, Target Selection Reward -- {}"
              .format(e+1, t, target.calculate_covered('mining'), target.calculate_covered('map'), target_selection_reward))
        print('***********')

        num_steps.append(t)
        num_average_s.append(t)
        if e < average_over:
            num_t = 0
            for i in range(e):
                num_t += num_average_s[i]
            num_t /= (e + 1)
            num_average_steps.append(num_t)
        else:
            num_average_steps.append(sum(num_average_s) / average_over)

        if e % average_over == 0:
            plt.plot(num_average_steps)
            plt.ylabel('Averaged Episode Steps')
            plt.xlabel('Episode')
            plt.savefig('Testing_results/' + str('Steps') + '/ddrqn_average_steps.png')
            plt.clf()



        if not target_cost and e % average_over == 0:
            selection_agent.save('select_model.{}'.format(e), 'select_target.{}'.format(e))
