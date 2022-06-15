import neat
import pygame
import numpy
import math
from mathfunctions import *
pygame.font.init()

WHITE = (255, 255, 255)
BLUE = (0, 0, 255)


class Car:
    def __init__(self, xpos, ypos, angle, pid, window):
        # Initial variables
        self.xpos = xpos
        self.ypos = ypos
        self.angle = angle
        self.width = 15
        self.height = 30
        self.speed = 0
        self.spritepath = "images/Topdown_vehicle_sprites_pack/Car.png"

        # Sets car sprite
        self.carimage = pygame.image.load(self.spritepath)
        self.scale = [self.width, self.height] #(self.height * self.carimageratio)]
        self.carimage = pygame.transform.scale(self.carimage, 
            (self.scale[0], self.scale[1]))
        self.carimage = pygame.transform.rotate(self.carimage, 180)
        self.carimage.convert()

        self.carmask = pygame.mask.from_surface(self.carimage)

        # detection lines = [[length, angle], [length, angle]]
        self.dlines = [[100, 0], [100, 45], [100, -45]]

        self.pid = pid
        self.font = pygame.font.SysFont('Comic Sans MS', 30)
        self.txt = "yey"
        self.numid = self.font.render(str(self.pid), False, (250, 250, 250))

        self.alive = True
        self.crashed = False
        self.overlapped = False

        self.fitness = 0

        rotated_image, self.new_rect = rot_center(self.carimage, self.angle, self.xpos, self.ypos)

        self.stagnantcar = [0, 0]

        self.linelist = []
        for dline in self.dlines:
            length = 0
            x = self.xpos
            y = self.ypos
            while window.get_at((int(x), int(y))) != (99, 99, 99) and length < dline[0]:
                length += 20
                x = int(self.xpos+xtrav(length, self.angle + dline[1]))
                y = int(self.ypos+ytrav(length, self.angle + dline[1]))
            self.linelist.append([length, x, y, dline])


    def draw(self, window):
        self.angle = angle_normalise(self.angle)
#        self.rotated_image, self.new_rect = rot_center(self.carimage, self.angle, self.xpos, self.ypos)

#        self.rotated_image = pygame.transform.rotate(self.carimage, self.angle)
#        self.new_rect = self.rotated_image.get_rect(center = 
#            self.carimage.get_rect(center = (self.xpos, self.ypos)).center)
#        self.carmask = pygame.mask.from_surface(self.rotated_image)

        window.blit(self.rotated_image, self.new_rect)

#        for i in self.linelist:
#            pygame.draw.line(window, WHITE, [self.xpos, self.ypos], [i[1], i[2]])


    # TODO: optimise this or tie it to NN
    def radar1(self, window, mask, frame):
        if self.speed > 0:
            self.speed -= 0.1
        self.rotated_image, self.new_rect = rot_center(self.carimage, self.angle, self.xpos, self.ypos)
        if frame % 2 == 0:
            self.linelist = []
            for dline in self.dlines:
                x = self.xpos
                y = self.ypos
                length = 0
                while mask.get_at((int(x), int(y))) == 0 and length < dline[0]:
#                while window.get_at((int(x), int(y))) != (99, 99, 99) and length < dline[0]:
                    length += 5
                    x = int(self.xpos+xtrav(length, self.angle + dline[1]))
                    y = int(self.ypos+ytrav(length, self.angle + dline[1]))
                self.linelist.append([length, x, y, dline])

    def radar2(self, window, mask, frame):
        if self.speed > 0:
            self.speed -= 0.1
        self.rotated_image, self.new_rect = rot_center(self.carimage, self.angle, self.xpos, self.ypos)
        if frame % 2 == 0:
            rec = 0
            for listline in self.linelist:
                length = listline[0] - 20
                if length < 0:
                    length = 0
                x = int(self.xpos+xtrav(length, self.angle + listline[3][1]))
                y = int(self.ypos+ytrav(length, self.angle + listline[3][1]))
                winat = window.get_at((int(x), int(y)))
                if winat == (99, 99, 99):
                    while winat == (99, 99, 99) and length > 0:
                        length -= 10
                        x = int(self.xpos+xtrav(length, self.angle + listline[3][1]))
                        y = int(self.ypos+ytrav(length, self.angle + listline[3][1]))
                        winat = window.get_at((int(x), int(y)))
                else:
                    while winat != (99, 99, 99) and length < listline[3][0]:
                        length += 10
                        x = int(self.xpos+xtrav(length, self.angle + listline[3][1]))
                        y = int(self.ypos+ytrav(length, self.angle + listline[3][1]))
                        winat = window.get_at((int(x), int(y)))
                self.linelist[rec] = ([length, x, y, listline[3]])
                rec += 1


    def momentum(self, far=0):
        distance = (self.speed)
        xmoved = xtrav(distance, self.angle)
        self.xpos += xmoved
        ymoved = ytrav(distance, self.angle)
        self.ypos += ymoved

    def forward(self, far=0):
        self.speed += 0.4
#        distance = far
#        xmoved = xtrav(distance, self.angle)
#        self.xpos += xmoved
#        ymoved = ytrav(distance, self.angle)
#        self.ypos += ymoved

    def left(self, ang=0):
        self.angle -= ang

    def right(self, ang=0):
        self.angle += ang


def rot_center(image, angle, x, y):
    rotated_image = pygame.transform.rotate(image, angle)
    new_rect = rotated_image.get_rect(center = image.get_rect(center = (x, y)).center)
    return rotated_image, new_rect


def colormask(image, mask_color):
    mask_image = image.convert()
    mask_image.set_colorkey(mask_color)
    mask = pygame.mask.from_surface(mask_image)
    mask.invert()
    return mask