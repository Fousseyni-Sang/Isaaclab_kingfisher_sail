#!/usr/bin/env python3

import os
import yaml
import math

def generate_experiments_yaml():
    # Define the experiments data

    v0s = [0.5]
    num_points = 6
    space = 2
    length = 8
    width = 4
    
    experiments = []
        
    points = [] 
    for v0 in v0s:
        # zigzag
        # for n in range(num_points//2):
        #     p1 = [(1+n)*space, space/space]
        #     p2 = [(1+n)*space + space/2, -space/space]
        #     points += [p1, p2]

        # square
        # p1 = [space, -space/2]
        # p2 = [2*space, -space/2]
        # p3 = [2*space, space/2]
        # p4 = [space, space/2]
        # points += [p1, p2, p3, p4]

        # line 
        # for n in range(num_points):
        #     points += [[n*space/2, 0]]

        # grid
        for w in range(width):
            for l in range(length):
                y = w * space
                if w % 2 == 0:
                    x = l*space
                else:
                    x = ((length-1) - l) * space # reverse direction
                points += [[x, y]]

    # for v0 in v0s:
    #     for n in range(num_points):
    #         points += [[math.cos(n*math.pi*2 / num_points) * 15 + 7.5, math.sin(n*math.pi*2 / num_points)*15 + 7.5]]
    
    # points2 = points[6:] + points[:6]
    experiments.append({'v0': v0, 'points': points})

    config_path = 'config'

    # Ensure the config directory exists
    if not os.path.exists(config_path):
        os.makedirs(config_path)

    # Path to the experiments.yaml file
    yaml_file_path = os.path.join(config_path, 'grid.yaml')

    # Write the experiments data to the YAML file
    with open(yaml_file_path, 'w') as yaml_file:
        yaml.safe_dump({'experiments': experiments}, yaml_file, default_flow_style=False)

    print(f"Generated {yaml_file_path}")

if __name__ == "__main__":
    generate_experiments_yaml()
