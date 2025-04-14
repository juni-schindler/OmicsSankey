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
