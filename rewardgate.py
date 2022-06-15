import pygame
from carplayer import colormask

YELLOW = (255,255,0)

class Rewardgate:
	def __init__(self, x1, y1, x2, y2, fitnessgain, window):
		self.x1 = x1
		self.y1 = y1
		self.x2 = x2
		self.y2 = y2
		self.fitnessgain = fitnessgain
		self.playerlisty = []

		self.rect = pygame.draw.line(window, YELLOW, [self.x1, self.y1], [self.x2, self.y2], 5)

		self.surface = window.subsurface(self.rect)
		self.mask = colormask(self.surface, YELLOW)


	def draw(self, window):
		pygame.draw.line(window, YELLOW, [self.x1, self.y1], [self.x2, self.y2], 5)
#		window.blit(self.mask.to_surface(), self.rect)

