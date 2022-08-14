from pickle import FALSE
from Environment.Snake_env import Point, Snake_Env
from utils.Parameter import Parameter
from Environment.Snake_env import Direction
import numpy as np
import torch
import random
from Deep_ExpectedSarsa.model_expected_sarsa import Linear_QNet, QTrainer
from collections import deque
from utils.utils import plot
import os


MAX_MEMORY = 100000
BATCH_SIZE = 1000
LR = 0.001
Train = False

class Snake_Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount factor
        self.memory = deque(maxlen=MAX_MEMORY) 
        self.model = Linear_QNet()
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    # get state
    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - Parameter.BLOCK_SIZE, head.y)
        point_r = Point(head.x + Parameter.BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - Parameter.BLOCK_SIZE)
        point_d = Point(head.x, head.y + Parameter.BLOCK_SIZE)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        point_l2 = Point(head.x - 2* Parameter.BLOCK_SIZE, head.y)
        point_r2 = Point(head.x + 2* Parameter.BLOCK_SIZE, head.y)
        point_u2 = Point(head.x, head.y - 2*Parameter.BLOCK_SIZE)
        point_d2 = Point(head.x, head.y + 2*Parameter.BLOCK_SIZE)
    

        # ktra ô chéo
        point_lc = Point(head.x - Parameter.BLOCK_SIZE, head.y - Parameter.BLOCK_SIZE)
        point_rc = Point(head.x + Parameter.BLOCK_SIZE, head.y + Parameter.BLOCK_SIZE)
        point_uc = Point(head.x + Parameter.BLOCK_SIZE, head.y - Parameter.BLOCK_SIZE)
        point_dc = Point(head.x - Parameter.BLOCK_SIZE, head.y + Parameter.BLOCK_SIZE)

        state = [
           # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),

            # Danger straight 2 step
            (dir_r and game.is_collision(point_r2)) or 
            (dir_l and game.is_collision(point_l2)) or 
            (dir_u and game.is_collision(point_u2)) or 
            (dir_d and game.is_collision(point_d2)),

            # Danger right 2 step
            (dir_u and game.is_collision(point_r2)) or 
            (dir_d and game.is_collision(point_l2)) or 
            (dir_l and game.is_collision(point_u2)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left 2 step
            (dir_d and game.is_collision(point_r2)) or 
            (dir_u and game.is_collision(point_l2)) or 
            (dir_r and game.is_collision(point_u2)) or 
            (dir_l and game.is_collision(point_d2)),

            # Danger chéo phải trên
            (dir_r and game.is_collision(point_rc)) or 
            (dir_l and game.is_collision(point_lc)) or 
            (dir_u and game.is_collision(point_uc)) or 
            (dir_d and game.is_collision(point_dc)),

            # Danger chéo phải dưới
            (dir_u and game.is_collision(point_rc)) or 
            (dir_d and game.is_collision(point_lc)) or 
            (dir_l and game.is_collision(point_uc)) or 
            (dir_r and game.is_collision(point_dc)),

            # Danger chéo trái trên
            (dir_d and game.is_collision(point_rc)) or 
            (dir_u and game.is_collision(point_lc)) or 
            (dir_r and game.is_collision(point_uc)) or 
            (dir_l and game.is_collision(point_dc)),

            # Danger chéo trái dưới
            (dir_d and game.is_collision(point_uc)) or 
            (dir_u and game.is_collision(point_rc)) or 
            (dir_r and game.is_collision(point_dc)) or 
            (dir_l and game.is_collision(point_uc)),


            # Move Direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food Location
            game.food.x < game.head.x,  # food is in left
            game.food.x > game.head.x,  # food is in right
            game.food.y < game.head.y,  # food is up
            game.food.y > game.head.y  # food is down
        ]
        return np.array(state, dtype=int)



    # train

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)


    def train_long_memory(self):
        if (len(self.memory) > BATCH_SIZE):
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reache

    def get_action_pretrained(self, state):
        checkpoint = ( os.getcwd() + '\\model\\v1.pth')
        final_move = [0,0,0]
        self.model.eval()
        self.model.load_state_dict(torch.load(checkpoint))
        with torch.no_grad():
            state0 = torch.tensor(state, dtype=torch.float)
            pred = self.model(state0)
        move = torch.argmax(pred).item()

        final_move[move] = 1
        return final_move
    
    def get_action(self, state):
        
        self.epsilon = 150 - self.n_games
        final_move = [0,0,0]

        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

    def train():
        plot_scores = []
        plot_mean_scores = []
        total_score = 0
        record = 0
        agent = Snake_Agent()
        game = Snake_Env()
        while True:
            # get old state
            state_old = agent.get_state(game)

            # get move
            if not Train:
                final_move = agent.get_action_pretrained(state_old)
            else: 
                final_move = agent.get_action(state_old)
            

            # perform move and get new state
            reward, done, score = game.step_env(final_move)
            state_new = agent.get_state(game)

   

            if  not Train:
                if done:
                    print('Game', agent.n_games, 'Score', score)
                    game.reset()
            else:
                         # train short memory
                agent.train_short_memory(state_old, final_move, reward, state_new, done)

            
                # remember
                agent.remember(state_old, final_move, reward, state_new, done)


                if done:
                    game.reset()
                    agent.n_games += 1
                    agent.train_long_memory()

                    if score > record:
                        record = score
                        agent.model.save()

                    print('Game', agent.n_games, 'Score', score, 'Record:', record)
                    
                    plot_scores.append(score)
                    total_score += score
                    mean_score = total_score / agent.n_games
                    plot_mean_scores.append(mean_score)
                    plot(plot_scores, plot_mean_scores)

                    



