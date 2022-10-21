from NavEnv import UAV, Sensor, Dynamics
from rlframework import Agent
from rlframework import plot_learning_curve

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.stats import rice
from mpl_toolkits import mplot3d



def plot_trajectory( trajectory, sensors, obstacle, ep_number, ep_step, obs_r, target_r, lim=30):

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:,2],'--bo')
    ax.plot(sensors[:,0], sensors[:,1], sensors[:,2],'--go')
    for obs in obstacle:
        ax.plot(obs[0], obs[1], obs[2], '--ro')
    
    
   

    
    ##for sensor in sensors:
        # rect = patches.Rectangle((sensor[0]-obs_radius, sensor[1]-obs_radius), 2*obs_radius, 2*obs_radius,
        #                          linewidth=1, edgecolor='k',  facecolor='g')
        # ax.add_patch(rect)
        ##ax.scatter(sensor[0], sensor[1], s=100, c='g')
        #circle = plt.Circle((sensor[0], sensor[1], sensor[2]), target_r, fill=False, edgecolor='g', linestyle='--')
        #ax.add_artist(circle)

    #for obs in obstacle:
     #   rect = plt.Rectangle((obs[0] - obs_r, obs[1] - obs_r), 2*obs_r, 2*obs_r,
                               #  fill = True, edgecolor='k', linewidth=1)
     #   ax.add_patch(rect)
    #ax.set_aspect('equal', adjustable='box')
    #ax.set_title(f"EPISODE {ep_number}, {ep_step} time steps")
    #ax.set_xlim(xmin=-0.0, xmax=lim+0)
    #ax.set_ylim(ymin=-0.0, ymax=lim+0)


if __name__ == '__main__':

    print("Main Dynamics is gonna be created")
    n_episodes = 1000
    SHOW_EVERY = 20
    EPISODE_TIME_LIMIT = 50
    OBS_RADIUS = 10
    height = 50
    Unit_size = 20
    TARGET_R = 10
    test_mode = False

    env = Dynamics(unit_size=Unit_size, uav_height=height, max_x=10, max_y=10, max_z=10, n_uav=1, n_sensor=1, n_obstacle=5,
                   obstacle_r=OBS_RADIUS, max_rangefinder=50, episode_time_limit=EPISODE_TIME_LIMIT, num_rangers=5
                   , target_r=TARGET_R)

    print("Main Action Space Shape", env.action_space.shape[0])
    agent = Agent(input_dims=env.state_space, env=env, n_actions=env.action_space.shape[0],
                  alpha=0.001, beta=0.001, batch_size=64, fc1=400, fc2=300, noise=0.2, tau=0.001)

    print("Main Agent is created")

    # env = gym.make('Pendulum-v0')
    # agent = Agent(input_dims=env.observation_space.shape, env=env, n_actions=env.action_space.shape[0],
    #               alpha=0.001, beta=0.001, batch_size=64, fc1=400, fc2=300, noise=0.2, tau=0.01)

    figure_file = 'plots/uav.png'
    best_score = -float('inf')
    score_history = []
    critic_losses = []
    actor_losses = []
    num_steps = 0
    observation = env.reset(schedule=[0], random=True) # UAV Class is called
    agent.initialize_networks(state=observation.reshape(1, -1))  # Create Neural Nets
    if test_mode:
        agent.load_models(number=None)

    locs = np.array([0, 10, 110, 155, 228, 240, 385, 390, 443, 532, 590, 345])
    
    for episode in range(n_episodes):
        # observation = env.reset(schedule=[0], random=True)
        # locs = np.array([357, 213, 518, 33, 248, 380, 525])
        # locs = np.random.choice(np.arange(0, 25 * 25), 1 + 5 + 1, replace=False)
        observation = env.reset(schedule=[0], random=True, locations=locs)
        done = False
        score, episode_step, energy = 0, 0, 0
        render = (episode % SHOW_EVERY == 0)

        while not done:
            episode_step += 1
            num_steps += 1
            print("####### MAIN EPISODE STEP NUMBER", episode, episode_step)
            # print("TIME STEP", num_steps)
            action = agent.choose_action(observation, test_mode)
            observation_, reward, done, trajectory, obs_centers, sensor_centers = env.step(action, episode_step)
            score += reward
            d_store = False  # if episode_step == EPISODE_TIME_LIMIT else done
            # print("MAIN Action", observation, action, reward, observation_)
            if not test_mode:
                agent.remember(observation, action, reward, observation_, d_store)
                agent.learn()
            observation = observation_

        score_history.append(score)

        if episode > (n_episodes - 20):
            plot_trajectory(trajectory, sensor_centers, obs_centers, episode, episode_step, obs_r=OBS_RADIUS,
                            target_r=TARGET_R, lim=10 * Unit_size)
        if not test_mode and episode % SHOW_EVERY == 0:
            agent.save_models()

        avg_score = np.mean(score_history[-SHOW_EVERY:])
        print('episode ', episode, "Time Step", episode_step, 'score %.1f' % score, 'avg score %.1f' % avg_score)
        print("#############################")

    print(score_history)
    plot_learning_curve(score_history, SHOW_EVERY, "Scores", figure_file)
    plt.show()