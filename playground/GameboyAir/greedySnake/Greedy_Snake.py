# Snake Tutorial Python
# Modified from https://github.com/GMfatcat/greedySnake

import random
import pygame
import sys
import tkinter as tk
from tkinter import messagebox

from finger_cursor.engine import Application, APPLICATION
from finger_cursor.utils import ExitException


class Cube(object):
    rows = 20
    w = 500

    def __init__(self, start, dirnx=1, dirny=0, color=(255, 0, 0)):
        self.pos = start
        self.dirnx = dirnx
        self.dirny = dirny
        self.color = color

    def move(self, dirnx, dirny):
        self.dirnx = dirnx
        self.dirny = dirny
        self.pos = (self.pos[0] + self.dirnx, self.pos[1] + self.dirny)  # change our position

    def draw(self, surface, eyes=False):
        dis = self.w // self.rows  # Width/Height of each Cube
        i = self.pos[0]  # Current row
        j = self.pos[1]  # Current Column

        pygame.draw.rect(surface, self.color, (i * dis + 1, j * dis + 1, dis - 2, dis - 2))
        # By multiplying the row and column value of our Cube by the width and height of each Cube we can determine where to draw it

        if eyes:  # Draws the eyes
            centre = dis // 2
            radius = 3
            circleMiddle = (i * dis + centre - radius, j * dis + 8)
            circleMiddle2 = (i * dis + dis - radius * 2, j * dis + 8)
            pygame.draw.circle(surface, (0, 0, 0), circleMiddle, radius)
            pygame.draw.circle(surface, (0, 0, 0), circleMiddle2, radius)


class Snake(object):
    body = []
    turns = {}

    def __init__(self, color, pos):
        self.color = color
        self.head = Cube(pos)  # The head will be the front of the snake
        self.body.append(self.head)  # We will add head (which is a Cube object)
        # to our body list

        # These will represent the direction our snake is moving
        self.dirnx = 0
        self.dirny = 0

    def move(self):
        global clock
        for event in pygame.event.get():
            if event.type == pygame.QUIT:  # Check if user hit the red x
                pygame.quit()
                sys.exit()

            keys = pygame.key.get_pressed()  # See which keys are being pressed

            for key in keys:  # Loop through all the keys
                if keys[pygame.K_LEFT]:
                    self.dirnx = -1
                    self.dirny = 0
                    self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]

                elif keys[pygame.K_RIGHT]:
                    self.dirnx = 1
                    self.dirny = 0
                    self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]

                elif keys[pygame.K_UP]:
                    self.dirnx = 0
                    self.dirny = -1
                    self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]

                elif keys[pygame.K_DOWN]:
                    self.dirnx = 0
                    self.dirny = 1
                    self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]

                elif keys[pygame.K_SPACE]:  # 案空白鍵可以暫停2秒
                    clock.tick(200)

                elif keys[pygame.K_ESCAPE]:
                    raise ExitException

        for i, c in enumerate(self.body):  # Loop through every Cube in our body
            p = c.pos[:]  # This stores the Cubes position on the grid
            if p in self.turns:  # If the Cubes current position is one where we turned
                turn = self.turns[p]  # Get the direction we should turn
                c.move(turn[0], turn[1])  # Move our Cube in that direction
                if i == len(self.body) - 1:  # If this is the last Cube in our body remove the turn from the dict
                    self.turns.pop(p)
            else:  # If we are not turning the Cube
                # If the Cube reaches the edge of the screen we will make it appear on the opposite side
                if c.dirnx == -1 and c.pos[0] <= 0:
                    c.pos = (c.rows - 1, c.pos[1])
                elif c.dirnx == 1 and c.pos[0] >= c.rows - 1:
                    c.pos = (0, c.pos[1])
                elif c.dirny == 1 and c.pos[1] >= c.rows - 1:
                    c.pos = (c.pos[0], 0)
                elif c.dirny == -1 and c.pos[1] <= 0:
                    c.pos = (c.pos[0], c.rows - 1)
                else:
                    c.move(c.dirnx, c.dirny)  # If we haven't reached the edge just move in our current direction

    def reset(self, pos):

        self.head = Cube(pos)
        self.body = []
        self.body.append(self.head)
        self.turns = {}
        self.dirnx = 0
        self.dirny = 1

    def addCube(self):
        tail = self.body[-1]  # 尾巴定義在身體的最後一個位置 [-1]是指最後位置
        dx, dy = tail.dirnx, tail.dirny

        # We need to know which side of the snake to add the Cube to.
        # So we check what direction we are currently moving in to determine if we
        # need to add the Cube to the left, right, above or below.
        if dx == 1 and dy == 0:
            self.body.append(Cube((tail.pos[0] - 1, tail.pos[1])))
        elif dx == -1 and dy == 0:
            self.body.append(Cube((tail.pos[0] + 1, tail.pos[1])))
        elif dx == 0 and dy == 1:
            self.body.append(Cube((tail.pos[0], tail.pos[1] - 1)))
        elif dx == 0 and dy == -1:
            self.body.append(Cube((tail.pos[0], tail.pos[1] + 1)))

        # We then set the Cubes direction to the direction of the snake.
        self.body[-1].dirnx = dx
        self.body[-1].dirny = dy

    def draw(self, surface):
        for i, c in enumerate(self.body):
            if i == 0:  # for the first Cube in the list we want to draw eyes
                c.draw(surface, True)  # adding the true as an argument will tell us to draw eyes
            else:
                c.draw(surface)  # otherwise we will just draw a Cube


def drawGrid(w, rows, surface):
    sizeBtwn = w // rows  # Gives us the distance between the lines
    x = 0  # Keeps track of the current x
    y = 0  # Keeps track of the current y
    for l in range(rows):  # We will draw one vertical and one horizontal line each loop
        x = x + sizeBtwn
        y = y + sizeBtwn

        pygame.draw.line(surface, (255, 255, 255), (x, 0), (x, w))
        pygame.draw.line(surface, (255, 255, 255), (0, y), (w, y))


def randomSnack(rows, item):
    positions = item.body

    while True:
        x = random.randrange(rows)
        y = random.randrange(rows)
        if len(list(filter(lambda z: z.pos == (x, y), positions))) > 0:
            # This wll check if the position we generated is occupied by the snake
            continue
        else:
            break
    return x, y


def message_box(subject, content):  # FIXME
    # root = tk.Tk()
    # root.attributes("-topmost", True)
    # root.withdraw()
    # messagebox.showinfo(subject, content)
    # try:
    #     root.destroy()
    # except:
    #     pass
    print(subject)
    print(content)
    raise ExitException


@APPLICATION.register()
class GreedySnake(Application):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.width = 500  # Width of our screen
        self.height = 500  # Height of our screen
        self.rows = 20  # Amount of rows

        self.win = pygame.display.set_mode((self.width, self.height))  # Creates our screen object

        self.s = Snake((255, 0, 0), (10, 10))  # Creates a snake object which we will code later

        self.clock = pygame.time.Clock()  # creating a clock object
        self.snack = Cube(randomSnack(self.rows, self.s), color=(0, 255, 0))
        self.playing_time = 1

    def loop(self):
        pygame.time.delay(200)  # This will delay the game so it doesn't run too quickly
        self.clock.tick(100)  # Will ensure our game runs at 10 FPS
        self.s.move()
        if self.s.body[0].pos == self.snack.pos:  # 如果被吃掉了 要產生新的點心
            self.s.addCube()
            self.snack = Cube(randomSnack(self.rows, self.s), color=(0, 255, 0))

        for x in range(len(self.s.body)):
            if self.s.body[x].pos in list(map(lambda z: z.pos, self.s.body[x + 1:])):
                print('Your score is : ', len(self.s.body) - 1)
                playt = str(self.playing_time)
                print('Times you play:  ' + playt)
                a = str(len(self.s.body) - 1)  # 頭一開始就有了 只算吃到幾個snack
                message_box('You lose >_<', ' Your score is : ' + a)
                self.s.reset((10, 10))
                self.playing_time += 1
                break

        self.redraw_window()  # This will refresh our screen

    def terminate(self):
        pass

    def redraw_window(self):
        self.win.fill((0, 0, 0))  # Fills the screen with black
        self.s.draw(self.win)
        self.snack.draw(self.win)  # NEW
        drawGrid(self.width, self.rows, self.win)  # Will draw our grid lines
        pygame.display.update()  # Updates the screen
