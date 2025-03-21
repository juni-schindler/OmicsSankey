import numpy as np
import json
from itertools import groupby
import random

# our file: contains matrix calculation
import algorithm

# out file: contains eigen calculation
import helper
import main
# ****************************************************************
# All outputs of this file are save in ../output/heur_case.txt
# ****************************************************************


# Set fixed variables like stated in paper
alpha1 = 0.1
N = 100
alpha2 = 0.1
M = 50
np.random.seed(26)

# ****************************************************************
# Processing data, adding dummny nodes, not part of the algorithm
# ****************************************************************
input_dir = "C:/Users/wasd2/Desktop/sankey_optimizer/example/input/heur_case.json"

data = helper.load_json(input_dir)
# specify which layer each node belongs to
level = {
    0: [
        "Primary Uranium",
        "Imports",
        "Fuel for Energy (in)",
        "Primary Oil",
        "Primary Natural Gas",
        "Primary Coal",
        "Primary Biomass",
        "Primary Hydroelectricity",
    ],
    1: [
        "Uranium Production",
        "Oil Production",
        "Natural Gas Production",
        "Coal Production",
        "Biofuel Production",
    ],
    2: [
        "Oil Domestic Use",
        "Natural Gas Domestic Use",
        "Coal Domestic Use",
        "Biofuel Domestic",
    ],
    3: ["Electricity Generation"],
    4: ["Electricity Domestic Use"],
    5: [
        "Non-Energy",
        "Commercial & Institutional",
        "Personal Transport",
        "Residential",
        "Industrial",
        "Freight Transportation",
    ],
    6: [
        "Exports",
        "Conversion Losses",
        "Useful Energy",
        "Fuel for Energy (out)",
        "Non-Energy Dummy End",
    ],
}
levelNumber = 7

# --------------------------------------------------------------------------------------
dummy_data = helper.dummy_data_preprocessing(data["links"], levelNumber, level)

# ****************************************************************
# Result of the combined method on this case obtained from the output graph
# ****************************************************************
heurOrder = [
    [8, 2, 1, 7, 6, 4, 3, 5],
    [6, 5, 4, 3, 2, 1, 7],
    [10, 13, 12, 5, 4, 14, 6, 3, 15, 7, 2, 16, 8, 1, 9, 11],
    [
        9,
        4,
        10,
        14,
        20,
        2,
        22,
        24,
        16,
        13,
        5,
        11,
        23,
        17,
        25,
        3,
        15,
        21,
        6,
        12,
        18,
        7,
        1,
        26,
        19,
        8,
    ],
    [
        10,
        4,
        11,
        16,
        22,
        2,
        24,
        26,
        18,
        15,
        5,
        12,
        25,
        19,
        27,
        3,
        17,
        23,
        6,
        13,
        20,
        7,
        1,
        14,
        9,
        28,
        21,
        8,
    ],
    [13, 7, 14, 18, 8, 15, 19, 9, 4, 1, 16, 10, 5, 6, 3, 2, 17, 20, 12, 11],
    [2, 5, 1, 4, 3],
]

HEUT_crossing = helper.calculate_crossings(
    heurOrder, dummy_data["nodes"], dummy_data["groupedLinks"]
)

print("***************************")
print("Combined method result as a baseline:")
print("Order from the graph produced by the combined method:")
print(heurOrder)
print("Weighted crossing: ", HEUT_crossing["weightedCrossing"])
print("Crossing: ", HEUT_crossing["crossing"])
# ----------------------------------------------------------------------------------------

nodes = []
result = {}
algo = "BC"
dummy_signal = True
cycle_signal = False
# result = main.run_method(algo, dummy_data['nodes'], dummy_data['layeredLinks'], dummy_data['addedLinks'], dummy_data['groupedLinks'], input_dir, levelNumber, alpha1, alpha2, N, M, dummy_signal)
result = main.run_method(
    algo,
    data,
    input_dir,
    levelNumber,
    alpha1,
    alpha2,
    N,
    M,
    dummy_signal,
    cycle_signal,
    level,
)
print(result)
