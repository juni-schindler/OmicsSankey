# ****************************************************************
# all outputs of this file are save in ../output/ilp_case.txt
# ****************************************************************
# our file: contains matrix calculation

# out file: contains eigen calculation
import json

import numpy as np

from sankey_optimizer import helper, main

# Set fixed variables like stated in paper
alpha1 = 0.01
N = 100
alpha2 = 0.1
M = 50

# ****************************************************************
# processing data, adding dummny nodes, not part of the algorithm
# ****************************************************************
input_dir = "input/ilp_case.json"
data = helper.load_json(input_dir)
# specify which layer each node belongs to
level = {
    0: ["Agriculture", "Waste", "Energy", "Industrial Processes", "Land Use Change"],
    1: [
        "Harvest / Management",
        "Deforestation",
        "Landfills",
        "Waste water - Other Waste",
        "Agriculture Soils",
        "Rice Cultivation",
        "Other Agriculture",
        "Livestock and Manure",
        "Electricity and heat",
        "Fugitive Emissions",
        "Transportation",
        "Industry",
        "Other Fuel Combustion",
    ],
    2: [
        "Coal Mining",
        "Machinery",
        "Pulp - Paper and Printing",
        "Air",
        "Unallocated Fuel Combustion",
        "Commercial Buildings",
        "T and D Losses",
        "Residential Buildings",
        "Food and Tobacco",
        "Iron and Steel",
        "Oil and Gas Processing",
        "Agricultural Energy Use",
        "Rail - Ship and Other Transport",
        "Road",
        "Aluminium Non-Ferrous Metals",
        "Other Industry",
        "Chemicals",
        "Cement",
    ],
    3: ["Carbon Dioxide", "HFCs - PFCs", "Methane", "Nitrous Oxide"],
}
levelNumber = 4
# add dummy nodes and links
np.random.seed(38)
dummy_data = helper.dummy_data_preprocessing(data["links"], levelNumber, level)


# ****************************************************************
# result of the BC method on the case obtained from the output graph
# ****************************************************************
bcOrder = [
    [2, 4, 3, 1, 5],
    [3, 12, 10, 6, 4, 2, 5, 17, 8, 16, 11, 14, 9, 15, 1, 7, 13],
    [
        16,
        15,
        7,
        17,
        10,
        14,
        2,
        1,
        19,
        13,
        18,
        9,
        12,
        20,
        8,
        11,
        22,
        24,
        5,
        23,
        27,
        6,
        28,
        26,
        21,
        3,
        25,
        4,
        29,
    ],
    [1, 2, 3, 4],
]

BC_crossing = helper.calculate_crossings(
    bcOrder, dummy_data["nodes"], dummy_data["groupedLinks"]
)

print("***************************")
print("BC method result as a baseline:")
print("Order from the graph produced by the BC method:")
print(bcOrder)
print("Weighted crossing: ", BC_crossing["weightedCrossing"])
print("Crossing: ", BC_crossing["crossing"])


# ****************************************************************
# result of the ILP method on the case obtained from the output graph
# ****************************************************************
ilpOrder = [
    [1, 5, 3, 2, 4],
    [1, 9, 8, 11, 13, 7, 4, 15, 16, 14, 17, 6, 3, 10, 12, 2, 5],
    [
        26,
        28,
        27,
        29,
        23,
        22,
        24,
        25,
        21,
        6,
        11,
        4,
        5,
        3,
        9,
        12,
        8,
        13,
        17,
        10,
        7,
        18,
        15,
        1,
        16,
        14,
        2,
        19,
        20,
    ],
    [4, 3, 2, 1],
]

ILP_crossing = helper.calculate_crossings(
    ilpOrder, dummy_data["nodes"], dummy_data["groupedLinks"]
)

print("***************************")
print("ILP method result as a baseline:")
print("Order from the graph produced by the ILP method:")
print(ilpOrder)
print("Weighted crossing: ", ILP_crossing["weightedCrossing"])
print("Crossing: ", ILP_crossing["crossing"])


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
with open("input/ilp_case_result.json", "w") as f:
    json.dump(result, f)
