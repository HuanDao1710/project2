from turtle import left
import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np
from utils.Parameter import Parameter

pygame.init() # khởi toạ ctirnh pygame
font = pygame.font.Font('data\\arial.ttf', 25)

param = Parameter()

class Direction(Enum): #liệt kê các hướng đi
    RIGHT = 1
    LEFT = 2 
    UP = 3
    DOWN = 4

Point = namedtuple('Point',['x','y'])

class Snake_Env:
    def __init__(self) -> None:
        self.w = param.Width
        self.h = param.Height
        self.wall_pos = []
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('SnakeAI')
        self.reset()
        self.clock = pygame.time.Clock()
    

    """
    def draw_grass(self):
        grass_color = (167,209,61) # màu cỏ
        cell_number = int(self.w / param.BLOCK_SIZE) #32
        cell_size = param.BLOCK_SIZE 
        for row in range(cell_number): 
            for col in range(cell_number):   
                grass_rect = pygame.Rect(col * cell_size,row * cell_size, cell_size, cell_size)
                pygame.draw.rect(self.display, grass_color, grass_rect)
    """
    
    
    def reset(self):
        # init game state
        self.direction = Direction.RIGHT
        self.food = None
        self.head = Point(self.w/2, self.h/2)
        
        self.snake = [self.head,
                      Point(self.head.x - param.BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2 * param.BLOCK_SIZE), self.head.y)]
        self.draw_wall()
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0




    def _place_food(self):
        
        while True:
            # sinh vị trí của food
            food = False
            x = random.randint(0, (self.w - param.BLOCK_SIZE )//param.BLOCK_SIZE )*param.BLOCK_SIZE 
            y = random.randint(0, (self.h - param.BLOCK_SIZE )//param.BLOCK_SIZE )*param.BLOCK_SIZE
            
            # kiểm tra food không trùng với tường
            for index, coord in enumerate(param.coords):
                if x == coord[0] and y == coord[1]:
                    break
                if index == len(param.coords) - 1:
                    food = True
            
            # lấy vị trí  food
            if food == True:
                self.food = Point(x, y)
                break
        
        # nếu food trùng với thân snake thì làm lại
        if self.food in self.snake:
            self._place_food()
       
    # vẽ food
    def draw_fruit(self):
        apple = pygame.image.load('data\\apple.png').convert_alpha()
        apple = pygame.transform.scale(apple, (param.BLOCK_SIZE, param.BLOCK_SIZE)) # convert do phan giai
        fruit_rect = pygame.Rect(self.food.x, self.food.y, param.BLOCK_SIZE, param.BLOCK_SIZE)# vi tri , kich thuoc cua food
        self.display.blit(apple,fruit_rect) # ve food vao


    # ve tuong
    def draw_wall(self):
        wall_color = (0, 0, 0)
        self.wall_pos = []
        self.bound = []
        wall = pygame.image.load('data\\wall.jpg').convert_alpha()
        wall = pygame.transform.scale(wall, (param.BLOCK_SIZE, param.BLOCK_SIZE))
        for coord in param.coords:
            x, y = coord
            wall_rect = pygame.Rect(x, y, param.BLOCK_SIZE, param.BLOCK_SIZE)
            pygame.draw.rect(self.display,wall_color,wall_rect)
            self.display.blit(wall, wall_rect)
            self.wall_pos.append([x, y])
        self.bound.append(np.array(self.wall_pos))
    

    def step_env(self,action):
        self.frame_iteration += 1

        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

                
        # 2. move   
        self._move(action) # update the head
        self.snake.insert(0, self.head)
        
        # 3. check if game over
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -25
            return reward, game_over, self.score


        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = +20
            self._place_food()
        else:
            self.snake.pop()
        

        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(param.FPS)

        
        # 6. return game over and score
        return reward, game_over, self.score


    def _update_ui(self):
        #draw snake
        # self.display.fill(BLACK)
        # draw snake
        self.display.fill((175, 215, 70))
        for index, pt in enumerate(self.snake):
            if index == 0:
                pygame.draw.rect(self.display, (0, 50, 255), pygame.Rect(pt.x, pt.y, param.BLOCK_SIZE, param.BLOCK_SIZE))
                pygame.draw.rect(self.display, (255, 255, 255), pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))
            else:
                pygame.draw.rect(self.display, param.BLUE1, pygame.Rect(pt.x, pt.y, param.BLOCK_SIZE, param.BLOCK_SIZE))
                pygame.draw.rect(self.display, param.BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))

        #pygame.draw.rect(self.display, param.RED, pygame.Rect(self.food.x, self.food.y, param.BLOCK_SIZE, param.BLOCK_SIZE))
        self.draw_fruit()
        self.draw_wall()
        text = font.render("Score: " + str(self.score), True, param.WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip() # cập nhật nội dung toàn bộ màn hình

    def is_collision(self, position=None):
        if position is None:
            position = self.head
        # hits boundary
        if position.x > self.w - param.BLOCK_SIZE or position.x < 0 or position.y > self.h - param.BLOCK_SIZE or position.y < 0:
            return True
        # hits wall

        for coord in param.coords:
            if position.x == coord[0]  and position.y == coord[1]:
                return True

        # hits itself
        if position in self.snake[1:]:
            return True
        return False


    def _move(self, action):
        # [straight, right, left]

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # right turn r -> d -> l -> u
        else: # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # left turn r -> u -> l -> d

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += param.BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= param.BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += param.BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= param.BLOCK_SIZE

        self.head = Point(x, y)
    

