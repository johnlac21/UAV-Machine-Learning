import numpy as np
from gym import spaces
from gym.utils import seeding
import random
from scipy.stats import rice


class UAV:

    def __init__(self, starting_x, starting_y, starting_z, phi=np.pi/4, max_speed=4, starting_energy=10,
                 slot_time=1):
        self.x = starting_x
        self.y = starting_y
        self.z = starting_z
        self.phi = phi
        self.speed = 0
        self.theta = np.pi/2
        self.max_speed = max_speed
        self.energy = starting_energy  # can be assumed it is +infinity
        self.slot_time = slot_time

        # Below parameters are for the energy consumption of the UAV. You can ignore them
        self.P0 = (0.012/8) * 1.225 * 0.05 * 0.79 * pow(400, 3) * pow(0.5, 3)
        self.Pi = (1 + 0.1) * pow(100, 1.5) / np.sqrt(2 * 1.225 * 0.79)
        self.lambda1 = pow(200, 2)
        self.lambda2 = 2 * pow(7.2, 2)
        self.lambda3 = 0.5 * 0.3 * 1.225 * 0.05 * 0.79
        self.zz=[]

    def power_consumption(self):  # need modification

        return self.P0 * (1 + 3 * pow(self.speed, 2) / self.lambda1) +\
               self.Pi * np.sqrt(np.sqrt(1 + pow(self.speed, 4)/pow(self.lambda2, 2)) - pow(self.speed, 2)/self.lambda2)+ \
               self.lambda3 * pow(self.speed, 3)

    def move(self, delta_v, delta_theta, delta_phi):
        self.x, self.y, self.z, self.speed, self.theta, self.phi = self.next_position(delta_v, delta_theta, delta_phi) #John
        p = self.power_consumption()
        self.energy -= p * self.slot_time
        return np.array([[self.x, self.y, self.z]]), p * self.slot_time
        # check before calling move function if it goes beyond the region borders/collide with obstacles

    def next_position(self, delta_v, delta_theta, delta_phi):
        speed_ = np.clip(self.speed + delta_v, 0, self.max_speed)  # Min Speed = 0?
        phi = (self.phi + delta_phi) % (2 * np.pi)
        theta = (self.theta + delta_theta) % (np.pi)
        print("theta = ", theta, self.z, np.cos(theta))
        x_ = self.x + speed_ * self.slot_time  * np.sin(theta) * np.cos(phi)
        y_ = self.y + speed_ * self.slot_time  * np.sin(theta) * np.sin(phi)
        z_ = self.z + speed_ * self.slot_time  * np.cos(theta)
        #x_ = self.x + speed_ * self.slot_time * np.cos(orient_)
        #y_ = self.y + speed_ * self.slot_time * np.sin(orient_)
        #z_ = self.z + speed_ * self.slot_time * np.sin(orient_) 
        #spherical coordinate system
        return x_, y_, z_,speed_, theta, phi

    def get_internal_state(self, x_lim, y_lim,z_lim):
        #return np.array([self.x/x_lim, self.y/y_lim, self.speed/self.max_speed, self.orientation/(2 * np.pi)])
        return np.array([self.speed/self.max_speed,self.theta/(np.pi), self.phi/(2 * np.pi)])

    def get_location(self):
        return np.array([self.x, self.y, self.z])


class Sensor:

    def __init__(self, x, y, z,starting_energy=1, slot_time=1, data=20, t_power=-40):
        self.x = 50
        self.y = 50
        self.z = 50
        self.energy = starting_energy
        self.slot_time = slot_time
        self.data = data
        self.power = t_power

    def get_location(self):
        return np.array([self.x, self.y, self.z])


class Dynamics:

    def __init__(self, unit_size=1., uav_height=10, obstacle_r=0.5, r_max=1., max_x=10, max_y=10, max_z=10, n_uav=1, n_sensor=1,
                 n_obstacle=1, max_rangefinder=0.5, num_rangers=3, episode_time_limit=100, target_r=20):
        self.R_MAX = r_max
        self.unit_size = unit_size      # Size of each Cell
        self.Mx = max_x     # Number of cells on X axis
        self.My = max_y     # Number of cells on Y axis
        self.Mz = max_z     # John
        self.n_uav = n_uav  # Number of UAVs
        self.n_sensor = n_sensor    # Number of Sensors
        self.n_obstacle = n_obstacle    # Number of Obstacles
        self.obstacle_radius = obstacle_r  # Obstacle's edge
        self.obs_centers = None
        self.sensor_centers = None
        self.sensor_covered = None
        self.schedule = None  # the order of targets to be visited
        self.current_target = None
        
        self.uav_centers = None
        self.uav_height = uav_height
        self.UAV_blob = None

        self.target_radius = target_r
        self.max_rangefinder = max_rangefinder
        self.num_rangers = num_rangers
        self.ranges = np.ones(num_rangers) * max_rangefinder
        self.state = None
        self.action_space = spaces.Box(low=np.array([-1, -1, -1]), high=np.array([1, 1, 1]), dtype=np.float32)#############################
        self.state_space = [11]
        self.time_limit = episode_time_limit
        self.time_counter = 0
        # self.observation_space
       
    def reset(self, schedule, random=True, locations=None):
        self.schedule = schedule
        self.locate_objects(random, locations) #UAV Class is called
        self.sensor_covered = -1 * np.ones(self.n_sensor)
        self.current_target = 0
        self.in_radius = 0
        self.set_rangers()
        # define state
        self.state = self.build_state()
        self.time_counter = 0
        self.trajectory = self.uav_centers

        return self.state
        # self.UAV_list = [UAV(center[0],center[1]) for center in self.uav_centers]

    def locate_objects(self, random, locations):
        # grid_map = np.zeros((self.My, self.Mx))
        if random:
            rnd_numbers = np.random.choice(np.arange(0, self.Mx * self.My),
                                           self.n_uav + self.n_obstacle + self.n_sensor, replace=False)
            rnd_numbers2 = np.random.choice(np.arange(0, self.Mx * self.My),
                                           self.n_uav + self.n_obstacle + self.n_sensor, replace=False)
        else:
            rnd_numbers = locations
        Ys = rnd_numbers // self.Mx
        Xs = rnd_numbers % self.Mx
        Ys_center = self.unit_size * (Ys + 1 / 2)
        Xs_center = self.unit_size * (Xs + 1 / 2)
        rand = np.random.randint(20,100,size=(7))
        Zs_center = rand
        #Zs_center = [0,0,0,0,0,0,0] # John
        # self.uav_centers = np.array([0.5, 0.5]) * self.unit_size  # UAV's Starting Point
        self.uav_centers = np.column_stack((Xs_center, Ys_center, Zs_center))[:self.n_uav]
        self.obs_centers = np.column_stack((Xs_center, Ys_center, Zs_center))[self.n_uav:self.n_uav + self.n_obstacle]
        self.sensor_centers = np.column_stack((Xs_center, Ys_center, Zs_center))[self.n_uav + self.n_obstacle:]
        # self.uav_centers = np.array([[0.5,0.5]])    ##UAV's Starting Point
        self.UAV_blob = UAV(self.uav_centers[0][0], self.uav_centers[0][1], 0, phi=np.pi/4, #self.UAV_height
                            max_speed=10)  # MAX Speed, UAV called
        self.Sensor_blobs = [Sensor(x=centers[0], y=centers[1], z=centers[2]) for centers in self.sensor_centers]
        #print("ENV-Locate Objects OBS SENSOR", self.obs_centers, self.sensor_centers)

    def build_state(self):             # Normalized State
        # a = (self.sensor_centers - self.uav_centers) / (self.unit_size * np.array([self.Mx, self.My]))
        i = self.schedule[self.choose_target()]
        a = (self.sensor_centers[i] - self.uav_centers) / (self.unit_size * np.array([self.Mx, self.My, self.Mz]))
        b = self.UAV_blob.get_internal_state(self.Mx*self.unit_size, self.My*self.unit_size, self.Mz * self.unit_size)
        inter = np.array([self.UAV_blob.x, self.UAV_blob.y, self.UAV_blob.z, self.UAV_blob.speed, self.UAV_blob.phi*180/np.pi,
                          self.ranges])
        self.raw_state = inter
        #return np.concatenate((a, self.sensor_covered, b, self.ranges/self.max_rangefinder), axis=None)
        return np.concatenate((a, b, self.ranges / self.max_rangefinder), axis=None)

    def set_rangers(self):
        x_uav = self.UAV_blob.x
        y_uav = self.UAV_blob.y
        or_uav = self.UAV_blob.phi
        angles = np.linspace(or_uav-np.pi/2, or_uav+np.pi/2, num=self.num_rangers)
        for i, angle in enumerate(angles):
            self.ranges[i] = self.find_single_range(x_uav, y_uav, angle, self.max_rangefinder)

    def find_single_range(self, x_uav, y_uav, angle, max_r):
        for j in np.arange(0.05, max_r, 0.5):  # Precision = 0.5 meter
            x = x_uav + j * np.cos(angle)
            y = y_uav + j * np.sin(angle)
            for obs in self.obs_centers:
                if abs(x - obs[0]) <= self.obstacle_radius and abs(y - obs[1]) <= self.obstacle_radius:
                    return j
        return max_r

    def step(self, action, ep_step):

        self.ep_step = ep_step
        action = action.numpy()
        delta_speed = action[0] * self.UAV_blob.max_speed  # Scale the output of the neural network
        delta_theta = action[1] * np.pi/2
        delta_phi = action[2] * np.pi
        print("RAW CURRENT STATE", self.raw_state)
        print("ACTION", delta_speed, delta_theta*180/np.pi)
        done = False
        # Move the UAV and watch for: 1-collision with obstacles, 2-out the border
        # print("radius", "theta", radius, theta)
        collision, border, target_reached, sensor_num = self.detect_event(delta_speed, delta_theta, delta_phi)
        print("Collision", "Border", "Target", collision, border, target_reached)
        sensor_reach = False

        trans_reward = 0
        energy_consumption = 0
        obs_penalty = 0
        free_reward = 0
        step_reward = -1
        data_reward = 0
        finish_reward = 0

        self.time_counter += 1
        #print("ENVIRONMENT TIMER", self.time_counter)

        if collision:
            obs_penalty = -20
        # elif border:
        #     border_penalty = 0
        else:

            new_pos, energy_consumption = self.UAV_blob.move(delta_speed, delta_theta, delta_phi)
            self.set_rangers()
            i = np.argmin(self.ranges)
            obs_penalty = -5 * np.exp(-1 * self.ranges[i])

            if self.ranges[2] == self.max_rangefinder:  # UAV heading towards a free-obstacle direction
                free_reward = 1

            trans_reward = self.transition_reward(new_pos) * 5

            if target_reached:
                self.sensor_covered[sensor_num] = 1
                self.current_target += 1

            target = (self.current_target == self.n_sensor)
            if target:
                self.current_target -= 1

            #target = np.all(self.sensor_covered == np.ones(self.n_sensor))

            self.uav_centers = new_pos
            if target:
                done = True
                finish_reward = 50

        if self.time_counter >= self.time_limit:
            done = True

        new_state = self.build_state()
        # You can remove the energy consumption penalty
        reward = trans_reward + obs_penalty + free_reward + step_reward + finish_reward # - 0.005 * energy_consumption

        self.trajectory = np.vstack((self.trajectory, self.raw_state[:3]))
        print("TOTAL REWARD", reward, "| TRANS Reward: ", trans_reward, "| OBS PEN", obs_penalty, "| FREE", free_reward)
        print("RAW NEW STATE", self.raw_state)
        print("NEW STATE ", new_state)
        return new_state, reward, done, self.trajectory, self.obs_centers, self.sensor_centers

    def detect_event(self, delta_v, delta_theta, delta_phi):
        collision = False
        border = False
        sensor_reach = False
        target = False

        x_, y_, z_ ,_, _,_ = self.UAV_blob.next_position(delta_v, delta_theta, delta_phi) #John
        speed_ = np.clip(self.UAV_blob.speed + delta_v, 0, self.UAV_blob.max_speed)  # Min Speed = 0
        orient_ = (self.UAV_blob.phi + delta_theta) % (2 * np.pi)
        x = self.UAV_blob.x
        y = self.UAV_blob.y

        # if x_ <= 0 or x_ >= self.Mx * self.unit_size \
        #         or y_ <= 0 or y_ >= self.My * self.unit_size:
        #     border = True
        # else:
        # for i, sensor in enumerate(self.sensor_centers):
        #     if self.sensor_covered[i]:
        #         continue
        #     if np.linalg.norm(np.array([x_, y_]) - sensor) <= self.UAV_blob.comm_radius():
        #     # if abs(new_uav_x - sensor[0]) <= self.UAV_blob.comm_radius() \
        #     #         and abs(new_uav_y - sensor[1]) <= self.UAV_blob.comm_radius():
        #         # Target Reached
        #         sensor_reach = True
        #         break


        # if distance <= self.UAV_blob.comm_radius():
        #     sensor_reach = True


        flag = False
        for i in range(len(self.obs_centers)):
            if(np.absolute(z_ - self.obs_centers[i][2]) < self.obstacle_radius):
                flag = True
                break

        obs_dis = self.find_single_range(x, y, orient_, speed_ * 1)
        if obs_dis != speed_ * 1 and flag:
            collision = True

        i = self.schedule[self.choose_target()]
        distance = np.linalg.norm(np.array([x_, y_, z_]) - self.sensor_centers[i])
        target_reached = 1 if distance < self.target_radius else 0

        #collision = False
        #for j in range(len(self.obs_centers)):
            #distance = np.linalg.norm(np.array([x_, y_, z_]) - self.obs_centers[j])
            #if(distance < self.obstacle_radius):
                #collision = True
                #break

        return collision, border, target_reached, i

    def transition_reward(self, new_pos=None):
        d_uav_sensor_1 = np.linalg.norm(self.uav_centers - self.sensor_centers, axis=1)
        d_uav_sensor_2 = np.linalg.norm(new_pos - self.sensor_centers, axis=1)
        print(new_pos, d_uav_sensor_1, d_uav_sensor_2)
        i = self.schedule[self.current_target]

        trans_reward = (d_uav_sensor_1[i] - d_uav_sensor_2[i])
        #decay = 1 - np.exp(-d_uav_sensor_1/self.UAV_blob.z)
        decay = self.in_radius
        print("Trans Reward", trans_reward)
        return 0.5 * trans_reward
        ####################FIX############

    def choose_target(self):
        # sorted_d = np.argsort(d_uav_sensor_1)
        # for i in sorted_d:
        #     if self.sensor_covered[i] == 0:
        #         break
        # for i in range(self.n_sensor):
        #     if self.sensor_covered[i] == -1:
        #         self.current_target = i
        #         return self.schedule[i]
        if self.current_target < self.n_sensor:
            return self.current_target
        else:
            return self.current_target - 1



