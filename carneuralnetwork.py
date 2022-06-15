import math
import os
import time
import random
from carplayer import *
from rewardgate import *
import visualize
import maxiNeuralReproduction as mnr
import maxiNeuralPopulation as mnp
# External libraries needed
import numpy
import pygame
# pip install neat-python
import neat
import wandb

global windowsize, playersource, targetpoint, target_fps, playerlist, netlist, genomelist
global runframelength, runtimelength, wandbon, bgimage, bgmask, rglist, gatelist, numofplayers
windowsize = [1000, 600]
playersource = [110, 300]
target_fps = 200
draw_fps = 30
runframelength = 300
runtimelength = 0
numofplayers = 0

# [[x1, y1, x2, y2, fitnessgain, order]]
rglist = [[65, 372, 155, 361, 100], [200, 444, 172, 529, 100], [264, 442, 342, 496, 100], 
[245, 340, 328, 302, 100], [174, 231, 258, 192, 100], [190, 68, 254, 142, 100], 
[293, 130, 286, 39, 100], [390, 40, 386, 130, 100], [521, 33, 525, 125, 100], 
[693, 36, 693, 129, 100], [794, 148, 863, 78, 100], [814, 179, 879, 250, 100], 
[750, 187, 750, 278, 100], [608, 189, 613, 282, 100], [532, 194, 548, 285, 100], 
[421, 289, 512, 311, 100], [521, 374, 446, 428, 100], [601, 390, 595, 481, 100], 
[739, 414, 731, 506, 100], [811, 415, 813, 509, 100]]

window = pygame.display.set_mode(size=(windowsize[0], windowsize[1]))

bgimage = pygame.image.load("images/track.png")
bgimage = pygame.transform.scale(bgimage, (windowsize[0], windowsize[1]))

bgmask = colormask(bgimage, (99, 99, 99))

wandbon = False
if wandbon == True:
    wandb.init(project="car-nn", entity="maxi25554")


def updatescreen():
    global playerlist, gatelist
    for rgate in gatelist:
        rgate.draw(window)
    for player in playerlist:
        player.draw(window)
    pygame.display.update()


def nneval(frame):
    global bgmask, numofplayers, playerlist, netlist, genomelist, window

    for i, player in enumerate(playerlist):
        if player.alive == True:
            player.radar1(window, bgmask, frame)

            nnevaloutput = netlist[i].activate((player.linelist[0][0], player.linelist[1][0], 
                player.linelist[2][0], player.speed))
#            nnevaloutput = netlist[i].activate((l1, l2, l3))
    #        player.txt = int(ang)

            x1, y1 = player.new_rect.topleft
            offset = (-x1, -y1)

            nnefar = 2
            nneang = 3


            if nnevaloutput[0] > 0:
                player.forward(nnefar*nnevaloutput[0])
            if nnevaloutput[1] > 0:
                player.left(nneang*nnevaloutput[1])
            if nnevaloutput[2] > 0:
                player.right(nneang*nnevaloutput[2])
            player.momentum()

            if player.carmask.overlap(bgmask, offset):
                player.alive = False
                player.crashed = True
                numofplayers -= 1
                break

            for gate in gatelist:
                if player not in gate.playerlisty:
                    x0, y0 = gate.rect.topleft
                    x1, y1 = player.new_rect.topleft
                    offset = (x0 - x1, y0 - y1)
                    if player.carmask.overlap(gate.mask, offset):
                        gate.playerlisty.append(player)
                        player.fitness += gate.fitnessgain
            
            if frame % 50 == 40:
                if player.xpos == player.stagnantcar[0] and (player.ypos == 
                    player.stagnantcar[1]) and player.fitness == 0:
                    player.alive = False
#                    remove(playerlist.index(player))
                    numofplayers -= 1 # len(playerlist)
    #                print (numofplayers)
                player.stagnantcar = [player.xpos, player.ypos]


def eval_genomes(genomes, config):
    global stats, generation, runtimelength, numofplayers, window
    global playerlist, genomelist, netlist, gatelist

    print ("genomelength: " + str(len(genomes)))
#    print (genomes)

    playerlist = []
    genomelist = []
    netlist = []
    gatelist = []

    print ("Runlength = " + str(runtimelength))

    for gate in rglist:
        gatelist.append(Rewardgate(gate[0], gate[1], gate[2], gate[3], gate[4], window))

    playeridmake = 0
    for genome_id, genome in genomes:
        playerlist.append(Car(playersource[0], playersource[1], 0, playeridmake, window))
        genomelist.append(genome)
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        netlist.append(net)
        genome.fitness = 0
        playeridmake += 1
    
    for player in playerlist:
        player.stagnantcar = [0, 0]

    running = True
    frame = 0
    runtimelength = 0
    prev_time = time.time()
    runtimelengthstart = time.time()
    frametime = 0
    numofplayers = len(playerlist)

    window.blit(bgimage, (0, 0))
    
    # Main sim loop
    while running == True:
        pygame.event.pump()

        # Runs neural network
#        nnevalstart = time.time()
        nneval(frame)
#        nnevalend = time.time()
#        nnevaldiff = (nnevalend - nnevalstart)
        # Updates window/draws stuff
        updatediff = 0
        if frametime >= (1/draw_fps):
            window.blit(bgimage, (0, 0))
#            updatestarttime = time.time()
            updatescreen()
            frametime = 0
#            frametime -= (1/draw_fps)
#            updateendtime = time.time()
#            updatediff = (updateendtime - updatestarttime)

#        if frame % int(target_fps/draw_fps) == 0:
#            updatescreen()

        # Manages framerate
        curr_time = time.time()
        diff = curr_time - prev_time
#        print (diff)
        frametime += diff
        delay = max(1.0/target_fps - diff, 0)
        if delay != 0:
            time.sleep(delay)
        prev_time = curr_time

        # Ends sim loop if frame high enough
        if frame == runframelength:
            frame = 0
            print ("Frame done")
            running = False
        else:
            frame += 1
        if numofplayers <= 0:
            print ("NOO")
            print ("Frame: " + str(frame))
            running = False

#        if frame % 5 == 0:
#        if updatediff != 0:
#            print ("Frame length: " + str(diff))
#            print ("Nneval length: " + str(nnevaldiff))
#            print ("Nneval percentage: " + str(nnevaldiff/diff*100))
#            print ("Update length: " + str(updatediff))
#            print ("Update percentage: " + str(updatediff/diff*100))

    runtimelengthend = time.time()
    runtimelength = runtimelengthend - runtimelengthstart

    fitnesslist = []
    bestfitness = 0
    dellist = []
    for i, player in enumerate(playerlist):
        playerfitness = fitnessfunc(player)
        genomes[i][1].fitness = playerfitness
        if playerfitness > bestfitness:
            bestfitness = playerfitness
        fitnesslist.append(playerfitness)
        if playerfitness == 0:
            dellist.append(genomes[i])

    # Removes stagnant genomes
    print ("oldgenlen: " + str(len(genomes)))
    if dellist != []:
        for i in dellist:
#            print ("popped: " + str(i))
            genomes.pop(genomes.index(i))

    print ("newgenomelength: " + str(len(genomes)))

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


def remove(index):
    global netlist, genomelist, playerlist
    netlist.pop(index)
    genomelist.pop(index)
    playerlist.pop(index)


def fitnessfunc(player):
    playerfitness = player.fitness
    return playerfitness


def run(config_path):
    global stats, generation
    config = neat.config.Config(
        neat.DefaultGenome,
        mnr.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )
    pop = mnp.Population(config)

    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)

    generation = 0
    winner = pop.run(eval_genomes, 100)

    print ("WINNN")
    print (winner)

    print ("Connections")
    print (winner.connections.values())

    # Turns winner neural net into diagrams/charts
    node_names = {-1: 'leftinp', -2: 'forwardinp', -3: 'rightinp', -4: 'speed', 0: 'forward', 1: 'left', 2: 'right'}
    visualize.draw_net(config, winner, True, node_names=node_names)
#    visualize.draw_net(config, winner, True, node_names=node_names, prune_unused=True)
#    visualize.plot_stats(stats, ylog=False, view=True)
#    visualize.plot_species(stats, view=True)


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'carnnconfig.txt')
    run(config_path)


"""
Car sprites: https://opengameart.org/content/free-top-down-car-sprites-by-unlucky-studio

"""