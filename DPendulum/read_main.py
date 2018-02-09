import math
import numpy as np
import pygame
from pygame import math as pmath
import drawer
import dynamics
import myio
import neuraldynamics

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
TIMESTEP = 0.01
FRAMERATE = 60
SPEEDUP = 1.0

def main():

    clock = pygame.time.Clock()

    size = [400, 300]
    screen = pygame.display.set_mode(size)
    screen.fill(WHITE)


    values, weights = myio.readNNet("out_NNstructure_init.txt")

    nnetwork = neuraldynamics.NNetwork(weights, values, 2, 5000)

    nnetwork.clear();

  #  for i in range(0,1000):
  #      nnetwork.calculate(TIMESTEP);

    #pendulum_drawer = drawer.PendulumDrawer(pmath.Vector2(200, 100), 75, 10)
    pendulum_drawer = drawer.DoublePendulumDrawer(pmath.Vector2(200, 100), 75, 75, 10, 10)

    wall_time = 0.0
    sim_time = 0.0

    pygame.font.init()
    myfont = pygame.font.SysFont('Arial', 20)

    pygame.init()

    while True:
        ev = pygame.event.poll()  # Look for any event
        if ev.type == pygame.QUIT:  # Window close button clicked?
           # myio.outABHistory("out.txt", pendulum.getHistory())
            break  # ... leave game loop

        if ((sim_time * 1000 - wall_time * SPEEDUP) < (pygame.time.get_ticks() - wall_time)):
            sim_time = nnetwork.calculate(TIMESTEP);

        if ((pygame.time.get_ticks() - wall_time) > 1.0 / FRAMERATE * 1000):
            state = nnetwork.getState()
            screen.fill(WHITE)
            pendulum_drawer.draw(screen, state[0], state[1])

            textsurface = myfont.render(str(sim_time), False, (0, 0, 0))
            screen.blit(textsurface, (0, 0))

            pygame.display.flip()
            wall_time = pygame.time.get_ticks()




    pygame.quit()     # Once we leave the loop, close the window.

main()