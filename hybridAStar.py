import math
import sys
import os
import heapq
import numpy as np
import matplotlib.pyplot as plt
from heapdict import heapdict
import scipy.spatial.kdtree as kd
from scipy.spatial.transform import Rotation
import CurvesGenerator.reeds_shepp as rsCurve

from copy import deepcopy
import numpy as np

class Auto:
    MAX_STEER_ANGLE = 0.6
    STEER_PRESION = 10
    WHEEL_BASE = 3.5
    AXLE_FRONT = 4.5
    AXLE_BACK = 1
    L = AXLE_FRONT + AXLE_BACK
    WIDTH = 3

    AUTO_RECT = [[AXLE_FRONT, WIDTH / 2.0], [AXLE_FRONT, -WIDTH / 2.0],
                [-AXLE_BACK, -WIDTH / 2.0], [-AXLE_BACK, WIDTH / 2.0],
                [AXLE_FRONT, WIDTH / 2.0]]
    
    CIRCLE_DIST = (AXLE_FRONT - AXLE_BACK) / 2.0
    CIRCLE_R = np.hypot((AXLE_FRONT + AXLE_BACK) / 2.0, WIDTH / 2.0)

    def __init__(self, x=0, y=0, yaw=0) -> None:
        self.x = x
        self.y = y
        self.yaw = yaw

    def draw(self, plt):
        Auto.draw_auto(plt, self.x, self.y, self.yaw)

    def move(self, step, steer):
        self.x, self.y, self.yaw = Auto.car_move(self.x, self.y, self.yaw, step, steer, Auto.WHEEL_BASE)

    @staticmethod
    def auto_commands():
        commands = []
        for steer_angle in np.linspace(-Auto.MAX_STEER_ANGLE, Auto.MAX_STEER_ANGLE, num=Auto.STEER_PRESION):
            commands.append([1.0, steer_angle])
            commands.append([-1.0, steer_angle])

        return commands


    @staticmethod
    def pi_2_pi(angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi
    
    @staticmethod
    def car_move(x, y, yaw, step, steer, L):
        x += step * np.cos(yaw)
        y += step * np.sin(yaw)
        yaw += Auto.pi_2_pi(step * np.tan(steer) / L)  # distance/2
        return x, y, yaw

    @staticmethod
    def draw_car_circle(plt:plt, x, y, yaw):
        cx = x + Auto.CIRCLE_DIST * np.cos(yaw)
        cy = y + Auto.CIRCLE_DIST * np.sin(yaw)
        circle = plt.Circle((cx, cy), Auto.CIRCLE_R, color='r', fill=False)
        plt.gcf().gca().add_artist(circle)

    @staticmethod
    def draw_car_arrow(plt:plt, x, y, yaw):
        plt.arrow(x, y, np.cos(yaw), np.sin(yaw), head_width=1.0)
    
    @staticmethod
    def draw_car_centroid(plt:plt, x, y):
        plt.scatter(x, y, s=2.0, c='r')
    
    @staticmethod
    def draw_car_shape(plt:plt, x, y, auto_rect, yaw):
        r = Rotation.from_euler('z', yaw, degrees=False)
        rot_matrix = r.as_matrix()[:2,:2]
        auto_rect_rotated = auto_rect.dot(rot_matrix.T)
        plt.plot(x + auto_rect_rotated[:, 0], y + auto_rect_rotated[:, 1])
    
    @staticmethod
    def draw_auto(plt:plt, x, y, yaw):
        Auto.draw_car_shape(plt, x, y, auto_rect=np.array(Auto.AUTO_RECT), yaw=yaw)
        Auto.draw_car_centroid(plt, x, y)
        Auto.draw_car_arrow(plt, x, y, yaw)
        Auto.draw_car_circle(plt, x, y, yaw)
    

class Cost:
    reverse = 10
    directionChange = 150
    steerAngle = 1
    steerAngleChange = 5
    hybridCost = 50

class MapParameters:
    def __init__(self, mapMinX, mapMinY, mapMaxX, mapMaxY, xyResolution, yawResolution, ObstacleKDTree, obstacleX, obstacleY):
        self.mapMinX = mapMinX               # map min x coordinate(0)
        self.mapMinY = mapMinY               # map min y coordinate(0)
        self.mapMaxX = mapMaxX               # map max x coordinate
        self.mapMaxY = mapMaxY               # map max y coordinate
        self.xyResolution = xyResolution     # grid block length
        self.yawResolution = yawResolution   # grid block possible yaws
        self.ObstacleKDTree = ObstacleKDTree # KDTree representating obstacles
        self.obstacleX = obstacleX           # Obstacle x coordinate list
        self.obstacleY = obstacleY           # Obstacle y coordinate list

def calculateMapParameters(obstacleX, obstacleY, xyResolution, yawResolution):
        
        # calculate min max map grid index based on obstacles in map
        mapMinX = round(min(obstacleX))
        mapMinY = round(min(obstacleY))
        mapMaxX = round(max(obstacleX))
        mapMaxY = round(max(obstacleY))

        # create a KDTree to represent obstacles
        ObstacleKDTree = kd.KDTree([[x, y] for x, y in zip(obstacleX, obstacleY)])

        return MapParameters(mapMinX, mapMinY, mapMaxX, mapMaxY, xyResolution, yawResolution, ObstacleKDTree, obstacleX, obstacleY)  

def map():
    # Build Map
    obstacleX, obstacleY = [], []

    for i in range(51):
        obstacleX.append(i)
        obstacleY.append(0)

    for i in range(51):
        obstacleX.append(0)
        obstacleY.append(i)

    for i in range(51):
        obstacleX.append(i)
        obstacleY.append(50)

    for i in range(51):
        obstacleX.append(50)
        obstacleY.append(i)
    
    for i in range(10,20):
        obstacleX.append(i)
        obstacleY.append(30) 

    for i in range(30,51):
        obstacleX.append(i)
        obstacleY.append(30) 

    for i in range(0,31):
        obstacleX.append(20)
        obstacleY.append(i) 

    for i in range(0,31):
        obstacleX.append(30)
        obstacleY.append(i) 

    for i in range(40,50):
        obstacleX.append(15)
        obstacleY.append(i)

    for i in range(25,40):
        obstacleX.append(i)
        obstacleY.append(35)

    # Parking Map
    # for i in range(51):
    #     obstacleX.append(i)
    #     obstacleY.append(0)

    # for i in range(51):
    #     obstacleX.append(0)
    #     obstacleY.append(i)

    # for i in range(51):
    #     obstacleX.append(i)
    #     obstacleY.append(50)

    # for i in range(51):
    #     obstacleX.append(50)
    #     obstacleY.append(i)

    # for i in range(51):
    #     obstacleX.append(i)
    #     obstacleY.append(40)

    # for i in range(0,20):
    #     obstacleX.append(i)
    #     obstacleY.append(30) 

    # for i in range(29,51):
    #     obstacleX.append(i)
    #     obstacleY.append(30) 

    # for i in range(24,30):
    #     obstacleX.append(19)
    #     obstacleY.append(i) 

    # for i in range(24,30):
    #     obstacleX.append(29)
    #     obstacleY.append(i) 

    # for i in range(20,29):
    #     obstacleX.append(i)
    #     obstacleY.append(24)

    return obstacleX, obstacleY

def run_hybrid_Astar(s, g, mapParameters, plt):

    # Compute Grid Index for start and Goal node
    sGridIndex = [round(s[0]), \
                  round(s[1])]
    gGridIndex = [round(g[0]), \
                  round(g[1])]
    
    x, y = [], []
    obstacleX, obstacleY = map()

    # heuristics_fun = lambda x, y : np.sum(np.abs(np.array(x) - np.array(y)))
    heuristics_fun = lambda x, y : np.sqrt(np.sum((np.array(x) - np.array(y)) ** 2))
    index_fun = lambda node : tuple(node[0])

    def conventional_is_vaild(x, y):
        if x<=mapParameters.mapMinX or x>=mapParameters.mapMaxX or \
            y<=mapParameters.mapMinY or y>=mapParameters.mapMaxY :
            return False
        for i in range(len(obstacleX)):
            if obstacleX[i] == x and obstacleY[i] == y:
                return False
        
        return True
            

    # Generate all Possible motion commands to car
    # motionCommand = [[1,0], [-1,0], [0, 1], [0, -1]]
    motionCommand = [[1,0], [-1,0], [0, 1], [0, -1], [1,1], [-1,-1], [1, -1], [-1, 1]]
    motionCommand.reverse()

    # Create start and end Node
    startNode = [sGridIndex, 0, heuristics_fun(sGridIndex, gGridIndex), None]
    goalNode = [gGridIndex, 0, 0, None]

    print(f"Start:{startNode}")
    print(f"End:{goalNode}")

    # Add start node to open Set
    openSet = {index_fun(startNode):startNode}
    closedSet = {}

    # Create a priority queue for acquiring nodes based on their cost's
    costQueue = heapdict()

    # Add start mode into priority queue
    costQueue[index_fun(startNode)] = startNode[2]

    counter = 0

    while len(openSet) != 0:
        node_index, node_cost = costQueue.popitem()
        cur_node = openSet[node_index]
        openSet.pop(node_index)
        # print(f"{cur_node}")

        closedSet[node_index] = cur_node

        if node_index == index_fun(goalNode):
            print("Path found!!")
            break

        for loc in motionCommand:
            new_loc = list(node_index)
            new_loc[0] += loc[0]
            new_loc[1] += loc[1]
            new_loc_index = tuple(new_loc)

            new_loc_h = heuristics_fun(new_loc_index, goalNode[0])
            new_loc_g = cur_node[1] + np.sqrt(np.sum((np.array(new_loc_index) - np.array(node_index)) ** 2))
            new_node = [new_loc, new_loc_g, new_loc_h, node_index]
            
            if conventional_is_vaild(new_loc[0], new_loc[1]):
                if new_loc_index not in closedSet:
                    if new_loc_index not in openSet:
                        openSet[new_loc_index] = new_node
                        costQueue[new_loc_index] = new_node[1] + new_node[2]
                    else:
                        if openSet[new_loc_index][1] > new_loc_g:
                            openSet[new_loc_index] = new_node
                            costQueue[new_loc_index] = new_node[1] + new_node[2]


    cur = goalNode
    while cur is not None:
        x.append(cur[0][0])
        y.append(cur[0][1])
        parent = closedSet[index_fun(cur)][3]
        if parent is not None:
            cur = closedSet[parent]
        else:
            cur = None

    x.reverse()
    y.reverse()

    return x, y, 0

def run_conventional_Astar(s, g, mapParameters, plt):

    # Compute Grid Index for start and Goal node
    sGridIndex = [round(s[0]), \
                  round(s[1])]
    gGridIndex = [round(g[0]), \
                  round(g[1])]
    
    x, y = [], []
    obstacleX, obstacleY = map()

    # heuristics_fun = lambda x, y : np.sum(np.abs(np.array(x) - np.array(y)))
    heuristics_fun = lambda x, y : np.sqrt(np.sum((np.array(x) - np.array(y)) ** 2))
    index_fun = lambda node : tuple(node[0])

    def conventional_is_vaild(x, y):
        if x<=mapParameters.mapMinX or x>=mapParameters.mapMaxX or \
            y<=mapParameters.mapMinY or y>=mapParameters.mapMaxY :
            return False
        for i in range(len(obstacleX)):
            if obstacleX[i] == x and obstacleY[i] == y:
                return False
        
        return True
            

    # Generate all Possible motion commands to car
    # motionCommand = [[1,0], [-1,0], [0, 1], [0, -1]]
    motionCommand = [[1,0], [-1,0], [0, 1], [0, -1], [1,1], [-1,-1], [1, -1], [-1, 1]]
    motionCommand.reverse()

    # Create start and end Node
    startNode = [sGridIndex, 0, heuristics_fun(sGridIndex, gGridIndex), None]
    goalNode = [gGridIndex, 0, 0, None]

    print(f"Start:{startNode}")
    print(f"End:{goalNode}")

    

    # Add start node to open Set
    openSet = {index_fun(startNode):startNode}
    closedSet = {}

    # Create a priority queue for acquiring nodes based on their cost's
    costQueue = heapdict()

    # Add start mode into priority queue
    costQueue[index_fun(startNode)] = startNode[2]

    counter = 0

    while len(openSet) != 0:
        node_index, node_cost = costQueue.popitem()
        cur_node = openSet[node_index]
        openSet.pop(node_index)
        # print(f"{cur_node}")

        closedSet[node_index] = cur_node

        if node_index == index_fun(goalNode):
            print("Path found!!")
            break

        for loc in motionCommand:
            new_loc = list(node_index)
            new_loc[0] += loc[0]
            new_loc[1] += loc[1]
            new_loc_index = tuple(new_loc)

            new_loc_h = heuristics_fun(new_loc_index, goalNode[0])
            new_loc_g = cur_node[1] + np.sqrt(np.sum((np.array(new_loc_index) - np.array(node_index)) ** 2))
            new_node = [new_loc, new_loc_g, new_loc_h, node_index]
            
            if conventional_is_vaild(new_loc[0], new_loc[1]):
                if new_loc_index not in closedSet:
                    if new_loc_index not in openSet:
                        openSet[new_loc_index] = new_node
                        costQueue[new_loc_index] = new_node[1] + new_node[2]
                    else:
                        if openSet[new_loc_index][1] > new_loc_g:
                            openSet[new_loc_index] = new_node
                            costQueue[new_loc_index] = new_node[1] + new_node[2]


    cur = goalNode
    while cur is not None:
        x.append(cur[0][0])
        y.append(cur[0][1])
        parent = closedSet[index_fun(cur)][3]
        if parent is not None:
            cur = closedSet[parent]
        else:
            cur = None

    x.reverse()
    y.reverse()

    return x, y, 0

def main():

    # Set Start, Goal x, y, theta
    global s, g
    s = [10, 10, np.deg2rad(0)]
    g = [25, 10, np.deg2rad(90)]


    # Get Obstacle Map
    obstacleX, obstacleY = map()

    # Calculate map Paramaters
    mapParameters = calculateMapParameters(obstacleX, obstacleY, 4, np.deg2rad(15.0))


    # Run Hybrid A*
    global x, y
    global updateX, updateY
    updateX = False
    updateY = False
    x, y, yaw = run_conventional_Astar(s, g, mapParameters, plt)


    def onclick(event):
        global x, y, yaw
        global updateX, updateY
        global s, g
        if updateX:
            g = [event.xdata, event.ydata, np.deg2rad(0)]
            x, y, yaw = run_conventional_Astar(s, g, mapParameters, plt)
            updateY = True
        else:
            s = [event.xdata, event.ydata, np.deg2rad(0)]
            updateX = True
        print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
            (event.button, event.x, event.y, event.xdata, event.ydata))
        

    fig = plt.figure()

    while True:
        
        fig.canvas.mpl_connect('button_press_event', onclick)
        x_temp = deepcopy(x)
        y_temp = deepcopy(y)
        yaw_temp = deepcopy(yaw)

        for k in range(len(x_temp)):
            if updateX and updateY:
                updateX = False
                updateY = False
                break
            plt.cla()
            plt.xlim(min(obstacleX), max(obstacleX)) 
            plt.ylim(min(obstacleY), max(obstacleY))
            plt.plot(obstacleX, obstacleY, "sk")
            plt.plot(x_temp, y_temp, linewidth=1.5, color='r', zorder=0)
            plt.scatter(x_temp[k], y_temp[k], color='b')

            plt.title("Hybrid A*")
            plt.pause(0.1)
    
    plt.show()

def main_2():

    car = Auto(0, 0, 45.0/180.0 * np.pi)

    for angle in np.arange(0, 2 * np.pi, 0.05):
        plt.cla()
        plt.xlim(-10, 10)
        plt.ylim(-10, 10)
        car.draw(plt)
        car.move(0.5, 1.0)
        plt.pause(0.001)
    plt.show()

if __name__ == '__main__':
    main_2()
