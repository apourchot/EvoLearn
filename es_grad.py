from copy import deepcopy
import argparse
import ray

import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

import gym.spaces
from tqdm import tqdm

from ES import sepCEM
from models import RLNN
from random_process import GaussianNoise, OrnsteinUhlenbeckProcess
from memory import Memory
from util import *

from typing import Union


# USE_CUDA = torch.cuda.is_available()
USE_CUDA = False
if USE_CUDA:
    FloatTensor = torch.cuda.FloatTensor
else:
    FloatTensor = torch.FloatTensor


class Actor(RLNN):

    def __init__(self, state_dim: int, action_dim: int, max_action: int, args):
        super(Actor, self).__init__(state_dim, action_dim, max_action)

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)

        if args.layer_norm:
            self.n1 = nn.LayerNorm(400)
            self.n2 = nn.LayerNorm(300)
        self.layer_norm = args.layer_norm

        self.optimizer = torch.optim.Adam(self.parameters(), lr=args.actor_lr)
        self.tau = args.tau
        self.discount = args.discount
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action

    def forward(self, x):
        if not self.layer_norm:
            x = torch.tanh(self.l1(x))
            x = torch.tanh(self.l2(x))
            x = self.max_action * torch.tanh(self.l3(x))

        else:
            x = torch.tanh(self.n1(self.l1(x)))
            x = torch.tanh(self.n2(self.l2(x)))
            x = self.max_action * torch.tanh(self.l3(x))

        return x

    def update(self, batch, critic, actor_t):
        # Sample replay buffer
        states, _, _, _, _ = batch

        # Compute actor loss
        if args.use_td3:
            actor_loss = -critic(states, self(states))[0].mean()
        else:
            actor_loss = -critic(states, self(states)).mean()

        # Optimize the actor
        self.optimizer.zero_grad()
        actor_loss.backward()
        self.optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(self.parameters(), actor_t.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data)


class Critic(RLNN):
    def __init__(self, state_dim, action_dim, max_action, args):
        super(Critic, self).__init__(state_dim, action_dim, 1)

        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

        if args.layer_norm:
            self.n1 = nn.LayerNorm(400)
            self.n2 = nn.LayerNorm(300)

        self.layer_norm = args.layer_norm
        self.optimizer = torch.optim.Adam(self.parameters(), lr=args.critic_lr)
        self.tau = args.tau
        self.discount = args.discount
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action

    def forward(self, x, u):

        if not self.layer_norm:
            x = F.leaky_relu(self.l1(torch.cat([x, u], 1)))
            x = F.leaky_relu(self.l2(x))
            x = self.l3(x)

        else:
            x = F.leaky_relu(self.n1(self.l1(torch.cat([x, u], 1))))
            x = F.leaky_relu(self.n2(self.l2(x)))
            x = self.l3(x)

        return x

    def update(self, batch, batch_size, actor_t, critic_t):

        # Sample replay buffer
        states, n_states, actions, rewards, dones = batch

        # Q target = reward + discount * Q(next_state, pi(next_state))
        with torch.no_grad():
            target_Q = critic_t(n_states, actor_t(n_states))
            target_Q = rewards + (1 - dones) * self.discount * target_Q

        # Get current Q estimate
        current_Q = self(states, actions)

        # Compute critic loss
        critic_loss = nn.MSELoss()(current_Q, target_Q)

        # Optimize the critic
        self.optimizer.zero_grad()
        critic_loss.backward()
        self.optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(self.parameters(), critic_t.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data)


class CriticTD3(RLNN):
    def __init__(self, state_dim, action_dim, max_action, args):
        super(CriticTD3, self).__init__(state_dim, action_dim, 1)

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

        if args.layer_norm:
            self.n1 = nn.LayerNorm(400)
            self.n2 = nn.LayerNorm(300)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 400)
        self.l5 = nn.Linear(400, 300)
        self.l6 = nn.Linear(300, 1)

        if args.layer_norm:
            self.n4 = nn.LayerNorm(400)
            self.n5 = nn.LayerNorm(300)

        self.layer_norm = args.layer_norm
        self.optimizer = torch.optim.Adam(self.parameters(), lr=args.critic_lr)
        self.tau = args.tau
        self.discount = args.discount
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.policy_noise = args.policy_noise
        self.noise_clip = args.noise_clip

    def forward(self, x, u):

        if not self.layer_norm:
            x1 = F.leaky_relu(self.l1(torch.cat([x, u], 1)))
            x1 = F.leaky_relu(self.l2(x1))
            x1 = self.l3(x1)

        else:
            x1 = F.leaky_relu(self.n1(self.l1(torch.cat([x, u], 1))))
            x1 = F.leaky_relu(self.n2(self.l2(x1)))
            x1 = self.l3(x1)

        if not self.layer_norm:
            x2 = F.leaky_relu(self.l4(torch.cat([x, u], 1)))
            x2 = F.leaky_relu(self.l5(x2))
            x2 = self.l6(x2)

        else:
            x2 = F.leaky_relu(self.n4(self.l4(torch.cat([x, u], 1))))
            x2 = F.leaky_relu(self.n5(self.l5(x2)))
            x2 = self.l6(x2)

        return x1, x2

    def update(self, batch, batch_size: int, actor_t: Actor, critic_t: Critic):

        # Sample replay buffer
        states, n_states, actions, rewards, dones = batch

        # Select action according to policy and add clipped noise
        noise = np.clip(np.random.normal(0, self.policy_noise, size=(
            batch_size, action_dim)), -self.noise_clip, self.noise_clip)
        n_actions = actor_t(n_states) + FloatTensor(noise)
        n_actions = n_actions.clamp(-max_action, max_action)

        # Q target = reward + discount * min_i(Qi(next_state, pi(next_state)))
        with torch.no_grad():
            target_Q1, target_Q2 = critic_t(n_states, n_actions)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = rewards + (1 - dones) * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self(states, actions)

        # Compute critic loss
        critic_loss = nn.MSELoss()(current_Q1, target_Q) + \
            nn.MSELoss()(current_Q2, target_Q)

        # Optimize the critic
        self.optimizer.zero_grad()
        critic_loss.backward()
        self.optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(self.parameters(), critic_t.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data)


@ray.remote
class CEMRLWorker:
    """
    This call is used for parallel actors evaluation and updates with ray.
    """
    def __init__(self, env, actor: Actor, critic: Union[Critic, CriticTD3]):
        self._env = env
        self._actor = actor
        self._critic = critic

    def evaluate(self, actor_params, n_episodes=1, random=False, noise=None):
        """
        Computes the score of an actor on a given number of runs,
        Returns the mean scores, the number of steps as well as
        the collected transitions.
        """

        self._actor.set_params(actor_params)

        if not random:
            def policy(state):
                state = FloatTensor(state.reshape(-1))
                action = self._actor(state).cpu().data.numpy().flatten()

                if noise is not None:
                    action += noise.sample()

                return np.clip(action, -max_action, max_action)

        else:
            def policy(state):
                return env.action_space.sample()

        scores = []
        transitions = []
        steps = 0

        for _ in range(n_episodes):

            score = 0
            obs = deepcopy(self._env.reset())
            done = False

            while not done:

                # get next action and act
                action = policy(obs)
                n_obs, reward, done, _ = self._env.step(action)
                done_bool = 0 if steps + \
                    1 == self._env._max_episode_steps else float(done)
                score += reward
                steps += 1

                # adding in memory
                transitions.append((obs, n_obs, action, reward, done_bool))
                obs = n_obs

                # reset when done
                if done:
                    self._env.reset()

            scores.append(score)

        return np.mean(scores), steps, transitions

    def update(self, batches, actor_params, critic_params, lr):
        """
        Updates the actor params weights on the batches on transitions.
        Returns the updated params.
        """
        self._actor.set_params(actor_params)
        actor_t = deepcopy(self._actor)
        self._critic.set_params(critic_params)
        self._actor.optimizer = torch.optim.Adam(actor.parameters(), lr=lr)

        for batch in tqdm(batches):
            self._actor.update(batch, critic, actor_t)

        return self._actor.get_params()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', default='train', type=str,)
    parser.add_argument('--env', default='HalfCheetah-v2', type=str)
    parser.add_argument('--start_steps', default=10000, type=int)

    # DDPG parameters
    parser.add_argument('--actor_lr', default=0.001, type=float)
    parser.add_argument('--critic_lr', default=0.001, type=float)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--reward_scale', default=1., type=float)
    parser.add_argument('--tau', default=0.005, type=float)
    parser.add_argument('--layer_norm', dest='layer_norm', action='store_true')

    # TD3 parameters
    parser.add_argument('--use_td3', dest='use_td3', action='store_true')
    parser.add_argument('--policy_noise', default=0.2, type=float)
    parser.add_argument('--noise_clip', default=0.5, type=float)
    parser.add_argument('--policy_freq', default=2, type=int)

    # Gaussian noise parameters
    parser.add_argument('--gauss_sigma', default=0.1, type=float)

    # OU process parameters
    parser.add_argument('--ou_noise', dest='ou_noise', action='store_true')
    parser.add_argument('--ou_theta', default=0.15, type=float)
    parser.add_argument('--ou_sigma', default=0.2, type=float)
    parser.add_argument('--ou_mu', default=0.0, type=float)

    # ES parameters
    parser.add_argument('--pop_size', default=10, type=int)
    parser.add_argument('--elitism', dest="elitism",  action='store_true')
    parser.add_argument('--n_grad', default=5, type=int)
    parser.add_argument('--sigma_init', default=1e-3, type=float)
    parser.add_argument('--damp', default=1e-3, type=float)
    parser.add_argument('--damp_limit', default=1e-5, type=float)
    parser.add_argument('--mult_noise', dest='mult_noise', action='store_true')

    # Training parameters
    parser.add_argument('--n_episodes', default=1, type=int)
    parser.add_argument('--max_steps', default=1000000, type=int)
    parser.add_argument('--mem_size', default=1000000, type=int)
    parser.add_argument('--n_noisy', default=0, type=int)

    # Testing parameters
    parser.add_argument('--filename', default="", type=str)
    parser.add_argument('--n_test', default=1, type=int)

    # misc
    parser.add_argument('--output', default='results/', type=str)
    parser.add_argument('--period', default=5000, type=int)
    parser.add_argument('--n_eval', default=10, type=int)
    parser.add_argument('--save_all_models',
                        dest="save_all_models", action="store_true")
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--seed', default=-1, type=int)
    parser.add_argument('--render', dest='render', action='store_true')

    args = parser.parse_args()
    args.output = get_output_folder(args.output, args.env)
    with open(args.output + "/parameters.txt", 'w') as file:
        for key, value in vars(args).items():
            file.write("{} = {}\n".format(key, value))

    # start ray
    ray.init()

    # environment
    env = gym.make(args.env)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = int(env.action_space.high[0])

    # memory
    memory = Memory(args.mem_size, state_dim, action_dim)

    # critic
    if args.use_td3:
        critic = CriticTD3(state_dim, action_dim, max_action, args)
        critic_t = CriticTD3(state_dim, action_dim, max_action, args)
        critic_t.load_state_dict(critic.state_dict())

    else:
        critic = Critic(state_dim, action_dim, max_action, args)
        critic_t = Critic(state_dim, action_dim, max_action, args)
        critic_t.load_state_dict(critic.state_dict())

    # actor
    actor = Actor(state_dim, action_dim, max_action, args)
    actor_t = Actor(state_dim, action_dim, max_action, args)
    actor_t.load_state_dict(actor.state_dict())

    workers = [CEMRLWorker.remote(env, actor, critic) for _ in range(args.pop_size)]

    # action noise
    if not args.ou_noise:
        a_noise = GaussianNoise(action_dim, sigma=args.gauss_sigma)
    else:
        a_noise = OrnsteinUhlenbeckProcess(
            action_dim, mu=args.ou_mu, theta=args.ou_theta, sigma=args.ou_sigma)

    if USE_CUDA:
        critic.cuda()
        critic_t.cuda()
        actor.cuda()
        actor_t.cuda()

    # CEM
    es = sepCEM(
        actor.get_size(),
        mu_init=actor.get_params(),
        sigma_init=args.sigma_init,
        damp=args.damp,
        damp_limit=args.damp_limit,
        pop_size=args.pop_size,
        antithetic=not args.pop_size % 2,
        parents=args.pop_size // 2,
        elitism=args.elitism
    )

    # training
    step_cpt = 0
    total_steps = 0
    actor_steps = 0

    df = pd.DataFrame(columns=[
        "total_steps",
        "average_score",
        "average_score_rl",
        "average_score_ea",
        "best_score"
    ])

    while total_steps < args.max_steps:

        fitness = []
        fitness_ = []
        es_params = es.ask(args.pop_size)

        # udpate the rl actors and the critic
        if total_steps > args.start_steps:

            # critic update
            for _ in tqdm(range(actor_steps)):
                batch = memory.sample(batch_size=args.batch_size)
                critic.update(batch, args.batch_size, actor, critic_t)

            # get new critic parameters
            critic_params = critic.get_params()

            # samples batches to train the actors
            batches_list = [
                [memory.sample(args.batch_size) for _ in range(actor_steps)]
                for _ in range(args.n_grad)
            ]

            # train actors in parallel
            updated_es_params = ray.get(
                [
                    workers[i].update.remote(
                        batches_list[i],
                        es_params[i],
                        critic_params,
                        args.actor_lr
                    )
                    for i in range(args.n_grad)
                ]
            )

            # update es parameters with new actors parameters
            for i in range(args.n_grad):
                es_params[i] = updated_es_params[i]

        actor_steps = 0

        # evaluate noisy actor(s) in parallel
        outs = ray.get(
            [
                workers[i].evaluate.remote(
                    es_params[i],
                    n_episodes=args.n_episodes,
                    noise=a_noise
                )
                for i in range(args.n_noisy)
            ]
        )

        # update replay buffer with collected transitions
        for f, steps, transitions in outs:

            for transition in transitions:
                memory.add(transition)

            actor_steps += steps
            prCyan('Noisy actor {} fitness:{}'.format(i, f))

        # evaluate all actors in parallel
        outs = ray.get(
            [
                workers[i].evaluate.remote(params, n_episodes=args.n_episodes)
                for i, params in enumerate(es_params)
            ]
        )

        # update replay buffer with collected transitions
        for f, steps, transitions in outs:

            for transition in transitions:
                memory.add(transition)

            actor_steps += steps
            fitness.append(f)

            # print scores
            prLightPurple('Actor fitness:{}'.format(f))

        # update es
        es.tell(es_params, fitness)

        # update step counts
        total_steps += actor_steps
        step_cpt += actor_steps

        # save stuff
        if step_cpt >= args.period:

            # evaluate mean actor over several runs. Memory is not filled
            # and steps are not counted
            f_mu, _, _ = ray.get([workers[0].evaluate.remote(
                es.mu,
                n_episodes=args.n_eval
            )])[0]

            prRed('Actor Mu Average Fitness:{}'.format(f_mu))

            df.to_pickle(args.output + "/log.pkl")

            average_score_half = np.partition(fitness, args.pop_size // 2 - 1)
            average_score_half = np.mean(average_score_half[args.pop_size // 2:])
            res = {
                "total_steps": total_steps,
                "average_score": np.mean(fitness),
                "average_score_half": average_score_half,
                "average_score_rl": np.mean(fitness[:args.n_grad]),
                "average_score_ea": np.mean(fitness[args.n_grad:]),
                "best_score": np.max(fitness),
                "mu_score": f_mu
            }

            if args.save_all_models:
                os.makedirs(
                    args.output + "/{}_steps".format(total_steps),
                    exist_ok=True
                )

                critic.save_model(
                    args.output + "/{}_steps".format(total_steps),
                    "critic"
                )

                actor.set_params(es.mu)
                actor.save_model(
                    args.output + "/{}_steps".format(total_steps),
                    "actor_mu"
                )
            else:
                critic.save_model(args.output, "critic")
                actor.set_params(es.mu)
                actor.save_model(args.output, "actor")

            df = df.append(res, ignore_index=True)
            step_cpt = 0
            print(res)

        print("Total steps", total_steps)
