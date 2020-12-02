import gym
import numpy as np
from random import randint
import matplotlib.pyplot as plt
import imageio
import sys
sys.setrecursionlimit(10**6) 


class SartorettiEnv:
    def __init__(self, env_size=None, obstacle_density=None, obs_radius=5, num_agents=1):
        self.episode = []
        if env_size is None:
            self.env_size = np.random.choice([10, 40, 70], p=[0.5, 0.25, 0.25])
        else:
            self.env_size = env_size
        if obstacle_density is None:
            self.obstacle_density = np.random.triangular(0, 0.33, 0.5)
        else:
            self.obstacle_density = obstacle_density
        self.num_agents = num_agents
        self.obs_radius = obs_radius
        self.generate_env()

    def get_padded_map(self, map, value):
        padded_shape = (map.shape[0] + 2 * self.obs_radius, map.shape[1] + 2 * self.obs_radius)
        result = np.ones(padded_shape) * value
        result[self.obs_radius:-self.obs_radius, self.obs_radius:-self.obs_radius] = np.copy(map)
        return result

    def get_map(self):
        map = np.zeros(self.obstacles.shape)
        map[self.obstacles != 0] = -1
        map[self.agents != 0] = self.agents[self.agents != 0]
        map[self.targets != 0] = self.targets[self.targets != 0] + self.num_agents
        return map

    def get_obs(self):
        obs = []
        for i in range(self.num_agents):
            obs.append(dict())
            obs[i]['obstacles'] = np.ones((2 * self.obs_radius + 1, 2 * self.obs_radius + 1))
            obs[i]['agents'] = np.zeros((2 * self.obs_radius + 1, 2 * self.obs_radius + 1))
            obs[i]['targets'] = np.zeros((2 * self.obs_radius + 1, 2 * self.obs_radius + 1))
            obs[i]['goal'] = np.zeros((2 * self.obs_radius + 1, 2 * self.obs_radius + 1))
            agent_coordinates = self.coordinates_2d[self.agents == (i + 1)][0]
            target_coordinates = self.coordinates_2d[self.targets == (i + 1)][0]
            # print(agent_coordinates - 5)
            obs[i]['obstacles'] = np.copy(
                self.get_padded_map(self.obstacles, 1)[agent_coordinates[0]:agent_coordinates[0] + 2 * self.obs_radius,
                agent_coordinates[1]:agent_coordinates[1] + 2 * self.obs_radius])
            obs[i]['agents'] = np.copy(
                self.get_padded_map(self.agents, 0)[agent_coordinates[0]:agent_coordinates[0] + 2 * self.obs_radius,
                agent_coordinates[1]:agent_coordinates[1] + 2 * self.obs_radius])
            for agent in obs[i]['agents'][obs[i]['agents'] != 0]:
                neighbour_target_coordinate = self.coordinates_2d[self.targets == agent][0]
                coord = [0, 0]
                if neighbour_target_coordinate[0] < agent_coordinates[0] - self.obs_radius:
                    coord[0] = 0
                elif neighbour_target_coordinate[0] > agent_coordinates[0] + self.obs_radius:
                    coord[0] = -1
                else:
                    coord[0] = neighbour_target_coordinate[0] - (agent_coordinates[0] - self.obs_radius)
                if neighbour_target_coordinate[1] < agent_coordinates[1] - self.obs_radius:
                    coord[1] = 0
                elif neighbour_target_coordinate[1] > agent_coordinates[1] + self.obs_radius:
                    coord[1] = -1
                else:
                    coord[1] = neighbour_target_coordinate[1] - (agent_coordinates[1] - self.obs_radius)
                # print(coord, obs[i]['targets'])
                if i + 1 == agent:
                    obs[i]['goal'][coord[0], coord[1]] = agent
                else:
                    obs[i]['targets'][coord[0], coord[1]] = agent
            if np.linalg.norm((target_coordinates - agent_coordinates)) != 0.0:
                obs[i]['goal_vector'] = (target_coordinates - agent_coordinates) / np.linalg.norm(
                    (target_coordinates - agent_coordinates))
            else:
                obs[i]['goal_vector'] = (target_coordinates - agent_coordinates)
            obs[i]['goal_magnitude'] = np.linalg.norm((target_coordinates - agent_coordinates))
            obs[i]['agents'][obs[i]['agents'] != 0] = 1
            obs[i]['targets'][obs[i]['targets'] != 0] = 1
            obs[i]['goal'][obs[i]['goal'] != 0] = 1
            # obs[i]['agents'][obs[i]['agents'] != 0] = 1
            # print(obs[i]['agents'])
            # print(obs[i]['targets'])
            # print(self.agents[5:-4, 5:-4])
            # print(self.targets[5:-4, 5:-4])
            # print(obs[i]['obstacles'])
        return obs

    def step(self, actions):
        if self.num_agents > 1:
            action_mapping = [[0, 0], [0, 1], [1, 0], [0, -1], [-1, 0]]
            rewards = np.zeros((self.num_agents,))

            for i in range(self.num_agents):
                agent_coordinates = self.coordinates_2d[self.agents == (i + 1)][0]
                target_coordinates = self.coordinates_2d[self.targets == (i + 1)][0]
                if actions[i] == 0:
                    if np.all(agent_coordinates == target_coordinates):
                        rewards[i] = 0.0
                    else:
                        rewards[i] = -0.3
                else:
                    direction = action_mapping[actions[i]]
                    if np.all(agent_coordinates + direction >= 0) and np.all(
                            agent_coordinates + direction < self.env_size):
                        if self.obstacles[(agent_coordinates + direction)[0], (agent_coordinates + direction)[1]] != 0 \
                                or self.agents[
                            (agent_coordinates + direction)[0], (agent_coordinates + direction)[1]] != 0:
                            rewards[i] = -2.0
                        else:
                            self.agents[(agent_coordinates + direction)[0], (agent_coordinates + direction)[1]] = i + 1
                            self.agents[agent_coordinates[0], agent_coordinates[1]] = 0
                            rewards[i] = -0.2
                    else:
                        rewards[i] = -2.0
            if np.all(rewards == 0.0):
                rewards = 20.0 * np.ones((self.num_agents,))
                if self.visualise_ep:
                    self.episode.append(self.get_map())
                return self.get_obs(), rewards, True
            if self.visualise_ep:
                self.episode.append(self.get_map())
            return self.get_obs(), rewards, False
        else:
            action_mapping = [[0, 0], [0, 1], [1, 0], [0, -1], [-1, 0]]
            rewards = 0
            agent_coordinates = self.coordinates_2d[self.agents == 1][0]
            target_coordinates = self.coordinates_2d[self.targets == 1][0]
            if actions == 0:
                if np.all(agent_coordinates == target_coordinates):
                    rewards = 0.0
                else:
                    rewards = -0.3
            else:
                direction = action_mapping[actions]
                if np.all(agent_coordinates + direction >= 0) and np.all(agent_coordinates + direction < self.env_size):
                    if self.obstacles[(agent_coordinates + direction)[0], (agent_coordinates + direction)[1]] != 0 \
                            or self.agents[(agent_coordinates + direction)[0], (agent_coordinates + direction)[1]] != 0:
                        rewards = -2.0
                    else:
                        self.agents[(agent_coordinates + direction)[0], (agent_coordinates + direction)[1]] = 1
                        self.agents[agent_coordinates[0], agent_coordinates[1]] = 0
                        rewards = -0.2
                else:
                    rewards = -2.0
            if rewards == 0.0:
                rewards = 20.0
                if self.visualise_ep:
                    self.episode.append(self.get_map())
                return self.get_obs(), rewards, True
            if self.visualise_ep:
                self.episode.append(self.get_map())
            return self.get_obs(), rewards, False

    def draw_ep(self, frame, i1):
        fig = plt.figure(figsize=(self.env_size, self.env_size))
        axs = [0, 0, 0]
        axs[0] = plt.subplot2grid((1, 1), (0, 0), colspan=2)
        axs[0].set_xlim(-0.5, self.env_size - 0.5)
        axs[0].set_ylim(-0.5, self.env_size - 0.5)
        for i in range(self.env_size):
            for j in range(self.env_size):
                if frame[i][j] == -1:
                    axs[0].scatter(i, j, marker='s', c='black')
        for i in range(self.env_size):
            for j in range(self.env_size):
                if frame[i][j] >= 1 and frame[i][j] <= self.num_agents:
                    axs[0].scatter(i, j, marker='^', c='black')
        for i in range(self.env_size):
            for j in range(self.env_size):
                if frame[i][j] > self.num_agents:
                    axs[0].scatter(i, j, marker='^', c='grey')

        fig.savefig(self.path + str(i1) + '.png', dpi=50)
        fig.clf()

    def make_gif(self, name):
        for i in range(len(self.episode)):
            self.draw_ep(self.episode[i], i)
        images = []
        for i in range(len(self.episode)):
            images.append(imageio.imread(self.path + str(i) + '.png'))
        imageio.mimsave(self.path + name + '.gif', images)

    def reset(self, seed=None, visualise_ep=False, path="/", name="test"):
        if len(self.episode) != 0:
            self.make_gif(self.name)
        self.episode = []
        self.visualise_ep = visualise_ep
        self.path = path
        self.name = name
        if seed is not None:
            np.random.seed(seed)
        else:
            np.random.seed(randint(0, 2 ** 32 - 1))
        self.generate_env()
        if self.visualise_ep:
            self.episode.append(self.get_map())
        return self.get_obs()

    def generate_env(self):
        self.steps = 0
        self.max_steps = 100
        self.obstacles = np.zeros((self.env_size, self.env_size))
        self.agents = np.zeros((self.env_size, self.env_size))
        self.targets = np.zeros((self.env_size, self.env_size))
        self.coordinates = np.array([[x, y] for y in range(self.env_size) \
                                     for x in range(self.env_size)])
        self.coordinates_2d = np.array([[[x, y] for y in range(self.env_size)] \
                                        for x in range(self.env_size)])
        obstacles = np.random.choice(range(len(self.coordinates)), size=int(self.env_size ** 2 * self.obstacle_density),
                                     replace=False)
        obstacles = self.coordinates[obstacles]
        self.obstacles[obstacles[:, 0], obstacles[:, 1]] = 1

        self.visited = [[False for i in range(self.env_size)] for j in range(self.env_size)]
        self.components = np.array([[-1 for i in range(self.env_size)] for j in range(self.env_size)])
        component_num = 0
        for i in range(self.env_size):
            for j in range(self.env_size):
                if not self.visited[i][j] and self.obstacles[i][j] == 0:
                    self.dfs(i, j, component_num)
                    component_num += 1

        for agent in range(self.num_agents):
            agent_coords = [0, 0]
            target_coords = [0, 0]
            while True:
                agent_coords = self.coordinates_2d[self.obstacles == 0][
                    np.random.choice(range(len(self.coordinates_2d[self.obstacles == 0])))]
                target_coords = self.coordinates_2d[self.components == \
                                                    self.components[agent_coords[0], agent_coords[1]]][
                    np.random.choice(range(len(self.coordinates_2d[self.components == \
                                                                   self.components[
                                                                       agent_coords[0], agent_coords[1]]])))]
                if (agent_coords[0] != target_coords[0] or agent_coords[1] != target_coords[1]) and \
                        self.agents[agent_coords[0], agent_coords[1]] == 0 and \
                        self.targets[target_coords[0], target_coords[1]] == 0:
                    break
            self.agents[agent_coords[0], agent_coords[1]] = agent + 1
            self.targets[target_coords[0], target_coords[1]] = agent + 1
        # self.get_obs()

    def dfs(self, x, y, component):
        self.visited[x][y] = True
        self.components[x][y] = component
        if x > 0:
            if not self.visited[x - 1][y] and self.obstacles[x - 1][y] == 0:
                self.dfs(x - 1, y, component)
        if x < self.env_size - 1:
            if not self.visited[x + 1][y] and self.obstacles[x + 1][y] == 0:
                self.dfs(x + 1, y, component)
        if y > 0:
            if not self.visited[x][y - 1] and self.obstacles[x][y - 1] == 0:
                self.dfs(x, y - 1, component)
        if y < self.env_size - 1:
            if not self.visited[x][y + 1] and self.obstacles[x][y + 1] == 0:
                self.dfs(x, y + 1, component)


if __name__ == '__main__':
    Env = SartorettiEnv(env_size=100, num_agents=100)
    Env.reset()
    from tqdm import trange
    for i in trange(1000):
        action = [randint(0, 4) for agent in range(100)]
        Env.step(action)