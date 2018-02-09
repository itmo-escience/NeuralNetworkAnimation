import pygame
import math
from pygame import math as pmath

BLACK   = (0, 0, 0)
RED     = (255, 0, 0)
BLUE    = (0, 0, 255)
GREEN   = (0, 255, 0)

class DoublePendulumDrawer:
    center_point = pmath.Vector2(0.0, 0.0)
    def __init__(self, O_point, len1, len2, size_ball1 = 5, size_ball2 = 5, line_color = BLACK, ball_color = RED):
        self.center_point = O_point

        self.len1 = len1
        self.len2 = len2

        self.ball1 = size_ball1
        self.ball2 = size_ball2

        self.line_color   = line_color
        self.ball_color   = ball_color


    def draw(self, screen, alpha, beta):
        x0 = int(self.center_point.x)
        y0 = int(self.center_point.y)

        x1 = int(x0 + self.len1 * math.sin(alpha))
        y1 = int(y0 + self.len1 * math.cos(alpha))

        x2 = int(x1 + self.len2 * math.sin(beta))
        y2 = int(y1 + self.len2 * math.cos(beta))

        pygame.draw.line(screen, self.line_color, [x0, y0], [x1, y1], 2)
        pygame.draw.line(screen, self.line_color, [x1, y1], [x2, y2], 2)

        pygame.draw.circle(screen, self.ball_color, [x1, y1], int(self.ball1))
        pygame.draw.circle(screen, self.ball_color, [x2, y2], int(self.ball2))


class PendulumDrawer:
    center_point = pmath.Vector2(0.0, 0.0)
    def __init__(self, O_point, len1, size_ball1 = 5, line_color = BLACK, ball_color = RED):
        self.center_point = O_point

        self.len1 = len1

        self.ball1 = size_ball1

        self.line_color   = line_color
        self.ball_color   = ball_color


    def draw(self, screen, alpha):
        x0 = int(self.center_point.x)
        y0 = int(self.center_point.y)

        x1 = int(x0 + self.len1 * math.sin(alpha))
        y1 = int(y0 + self.len1 * math.cos(alpha))

        pygame.draw.line(screen, self.line_color, [x0, y0], [x1, y1], 2)

        pygame.draw.circle(screen, self.ball_color, [x1, y1], int(self.ball1))

