from distangles import *
import neat
import pygame
import numpy
import math
pygame.font.init()

WHITE = (255, 255, 255)
BLUE = (0, 0, 255)


class Ball:
    def __init__(self, xpos, ypos, angle, pid):
        self.xpos = xpos
        self.ypos = ypos
        self.angle = angle
        self.width = 30
        self.height = 30

        # detection lines = [[length, angle], [length, angle]]
        self.dlines = [[150, 0], [150, 45], [150, -45]]

        self.pid = pid
        self.font = pygame.font.SysFont('Comic Sans MS', 30)
        self.txt = "yey"
        self.numid = self.font.render(str(self.pid), False, (250, 250, 250))


    def draw(self, window):
        pygame.draw.circle(window, BLUE, [self.xpos, self.ypos], self.width)

        self.angle = angle_normalise(self.angle)

#        self.numid = self.font.render(str(int(self.txt)), False, (250, 250, 250))

        window.blit(self.numid, (self.xpos, self.ypos))

        self.linelist = []
        for i in self.dlines:
            pygame.draw.line(window, WHITE, [self.xpos, self.ypos], 
                [(self.xpos+xtrav(i[0], i[1]+self.angle)), (self.ypos+ytrav(i[0], i[1]+self.angle))])


    def move(self, distance=0, angle=0):
        xmoved = xtrav(distance, self.angle)
        self.xpos += xmoved
        ymoved = ytrav(distance, self.angle)
        self.ypos += ymoved
        self.angle += angle


    def neurmove(self, forward, left, right):
        if forward == 1:
            distance = 1
            xmoved = xtrav(distance, self.angle)
            self.xpos += xmoved
            ymoved = ytrav(distance, self.angle)
            self.ypos += ymoved

        if left == 1:
            self.angle -= 1

        if right == 1:
            self.angle += 1


    def forward(self, far=0):
        distance = far*12
        xmoved = xtrav(distance, self.angle)
        self.xpos += xmoved
        ymoved = ytrav(distance, self.angle)
        self.ypos += ymoved

    def left(self, ang=0):
        self.angle -= ang*3

    def right(self, ang=0):
        self.angle += ang*3



def angle_between(p1, p2):
    angle = math.atan2(p2[1] - p1[1], p2[0] - p1[0]) * 180 / numpy.pi+180
    return angle


def angle_normalise(angle):
    if angle < 0:
        angle += 360
    elif angle >= 360:
        angle -= 360
    return angle