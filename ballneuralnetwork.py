import math
import os
import time
import random
from ballplayer import *
import visualize
# External libraries needed
import numpy
import pyglet
# pip install neat-python
import neat
import wandb

global windowsize, playersource, targetpoint, window, frame, target_fps 
global runframelength, runtimelength, wandbon
windowsize = [1000, 600]
playersource = [50, 50]
targetpoint = [600, 100]
frame = 0
target_fps = 1200
draw_fps = 60
runframelength = 1200
runtimelength = 0

wandbon = True
if wandbon == True:
    wandb.init(project="test-nn", entity="maxi25554")

window = pygame.display.set_mode(size=(windowsize[0], windowsize[1]))


def updatescreen(playerlist):
    window.fill([0,0,0])

    for player in playerlist:
        player.draw(window)
    pygame.draw.circle(window, [255, 255, 255], targetpoint, 20)
    pygame.display.update()


def nneval(playerlist, netlist):
    global nnevaloutput, targetpoint

    for i, player in enumerate(playerlist):
        distance = math.dist([player.xpos, player.ypos], targetpoint)
        ang = angle_between([player.xpos, player.ypos], targetpoint)+player.angle-90
        ang = angle_normalise(ang)
        ang = angle_normalise(ang)
        ang -= 180

#        nnevaloutput = netlist[i].activate((player.xpos, player.ypos, (player.angle%360)))
        nnevaloutput = netlist[i].activate((distance, ang))
#        player.txt = int(ang)

        player.forward(nnevaloutput[0])
        if nnevaloutput[1] >= 0.5:
            player.left(nnevaloutput[1]-0.5)
        if nnevaloutput[2] >= 0.5:
            player.right(nnevaloutput[2]-0.5)


def eval_genomes(genomes, config):
    global stats, targetpoint, playersource, generation, runtimelength, frame

    playerlist = []
    genomelist = []
    netlist = []

    # Randomises point of origin and target
    random.seed()
    targetpoint[0] = random.randint(100, 900)
    targetpoint[1] = random.randint(100, 500)
    playersource[0] = random.randint(100, 900)
    playersource[1] = random.randint(100, 500)

    print ("Runlength = " + str(runtimelength))

    playeridmake = 0
    for genome_id, genome in genomes:
        playerlist.append(Ball(playersource[0], playersource[1], 0, playeridmake))
        genomelist.append(genome)
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        netlist.append(net)
        genome.fitness = 0
        playeridmake += 1

    running = True
    frame = 0
    runtimelength = 0
    prev_time = time.time()
    runtimelengthstart = time.time()

    # Main sim loop
    while running == True:
        pygame.event.pump()
        # Runs neural network
        nneval(playerlist, netlist)
        # Updates window/draws stuff
        if frame % int(target_fps/draw_fps) == 0:
            updatescreen(playerlist)
        # Manages framerate
        curr_time = time.time()
        diff = curr_time - prev_time
        delay = max(1.0/target_fps - diff, 0)
        if delay != 0:
            time.sleep(delay)
        prev_time = curr_time
        # Ends sim loop if frame high enough
        if frame == runframelength:
            frame = 0
            running = False
        else:
            frame += 1

    runtimelengthend = time.time()
    runtimelength = runtimelengthend - runtimelengthstart

    sourcetargetdistance = math.dist(playersource, targetpoint)
    fitnesslist = []
    bestfitness = 0
    for i, player in enumerate(playerlist):
        playerfitness = ballfitnessfunc(player, sourcetargetdistance)
        genomes[i][1].fitness = playerfitness
        if playerfitness > bestfitness:
            bestfitness = playerfitness
        fitnesslist.append(playerfitness)

    # Finds mean fitness across generation
    fitnesssum = 0
    for i in fitnesslist:
        fitnesssum += i
    meanfitness = fitnesssum / (len(fitnesslist))

    if wandbon == True:
        wandb.log({"bestfitness": bestfitness, "meanfitness": meanfitness, 
            "runtimelength": runtimelength})

    pygame.event.wait()

    generation += 1
#    time.sleep(20)


def ballfitnessfunc(player, sourcetargetdistance):
    distance = math.dist([player.xpos, player.ypos], targetpoint)
    distanceadj = distance/sourcetargetdistance
    if distanceadj == 1:
        playerfitness = 0.1
    else:
        playerfitness = 100/(numpy.sqrt((distanceadj*100)+1))
    return playerfitness


def run(config_path):
    global stats, generation
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )
    pop = neat.Population(config)

    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)

    generation = 0
    winner = pop.run(eval_genomes, 300)

    print ("WINNN")
    print (winner)

    print ("Connections")
    print (winner.connections.values())

    # Turns winner neural net into diagrams/charts
    node_names = {-1: 'distance', -2: 'angle', 0: 'forward', 1: 'left', 2: 'right'}
    visualize.draw_net(config, winner, True, node_names=node_names)
#    visualize.draw_net(config, winner, True, node_names=node_names, prune_unused=True)
#    visualize.plot_stats(stats, ylog=False, view=True)
#    visualize.plot_species(stats, view=True)


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'ballnnconfig.txt')
    run(config_path)