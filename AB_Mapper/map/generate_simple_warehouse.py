import yaml
import numpy as np

def write_yaml_file(matrix, file_path='custom_map.yaml'):
    obstacles = []
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] == 1:
                obstacles.append((i, j))
    dimensions = [matrix.shape[0], matrix.shape[1]]
    data = {'map': {'dimensions': dimensions, 'obstacles': obstacles}}
    with open(file_path, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)

def get_warehouse_obs():
    return np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0],
                    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                    [0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0],
                    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                    [0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0],
                    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                    [0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    ]).transpose()

def get_open_list():    
    open_list = [[3, 0],
                [4, 0],
                [5, 0],
                [6, 0],
                [7, 0],
                [8, 0],
                [9, 0],
                [10, 0],
                [11, 0],
                [12, 0], 
                [3, 14],
                [4, 14],
                [5, 14],
                [6, 14],
                [7, 14],
                [8, 14],
                [9, 14],
                [10, 14],
                [11, 14],
                [12, 14],
                [2, 4],
                [2, 6],
                [2, 8],
                [2, 10],
                [4, 4],
                [4, 6],
                [4, 8],
                [4, 10],
                [6, 4],
                [6, 6],
                [6, 8],
                [6, 10],
                [8, 4],
                [8, 6],
                [8, 8],
                [8, 10],
                [10, 4],
                [10, 6],
                [10, 8],
                [10, 10],
                [12, 4],
                [12, 6],
                [12, 8],
                [12, 10]]
    
    return open_list

if __name__ == "__main__":
    matrix = get_warehouse_obs()
    write_yaml_file(matrix, 'simple_warehouse.yaml')