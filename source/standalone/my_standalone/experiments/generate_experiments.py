#!/usr/bin/env python3

import os
import yaml

def generate_experiments_yaml():
    # Define the experiments data

    v0s = [0.5]
    dists = [3, 6, 9]
    bearings = [0, -30, 30, -45, 45]

    experiments = []
    for v0 in v0s:
        for dist in dists:
            for bearing in bearings:
                print(f"{{'v0': {v0}, 'dist': {dist}, 'bearing': {bearing}}},")
                experiments.append({'v0': v0, 'dist': dist, 'bearing': bearing})

    
    config_path = 'config'

    # Ensure the config directory exists
    if not os.path.exists(config_path):
        os.makedirs(config_path)

    # Path to the experiments.yaml file
    yaml_file_path = os.path.join(config_path, 'experiments.yaml')

    # Write the experiments data to the YAML file
    with open(yaml_file_path, 'w') as yaml_file:
        yaml.safe_dump({'experiments': experiments}, yaml_file, default_flow_style=False)

    print(f"Generated {yaml_file_path}")

if __name__ == "__main__":
    generate_experiments_yaml()