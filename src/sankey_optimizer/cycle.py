import numpy as np
import json
from itertools import groupby
import algorithm
import helper
import matplotlib.pyplot as plt
import main

# ****************************************************************
# two output pictures of this file are saved as ../output/dumbbell.png and ../output/box.pnd=g
# ****************************************************************


# Set fixed variables like stated in paper
alpha1 = 0.01
N = 100
alpha2 = 0.01
M = 50

# ****************************************************************
# Functions for generating the cycle case
# ****************************************************************
levelNumber = 4
levels = {
    "-1": 3,
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 0,
}


# ****************************************************************
# Looping 20 test cases
# ****************************************************************
# all weighted crossing are recorded here
levelSize = [5, 17, 29, 4]
levelNumber = 4
totalSize = 100
stage1 = []
stage2 = []
np.random.seed(5)
for caseid in range(0, 20):
    # ****************************************************************
    # processing data
    # ****************************************************************
    data = helper.getOptimalCase(levels, levelSize, levelNumber, totalSize)
    edgeSum = data["edge"]
    """links = data['links']
    nodes = data['nodes']
    completeLinks = data['completeLinks']
    edgeSum = data['edge']
    addedLinks = completeLinks
    addedLinks.sort(key=lambda content: content['source'])
    groups3 = groupby(addedLinks, lambda content: content['source'])
    groupedLinks = {}
    for source, linkss in groups3:
        groupedLinks[source] = list(linkss)"""

    input_dir = " "
    result = {}
    algo = "BC"
    dummy_signal = False
    cycle_signal = True
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
        levels,
    )

    stage1.append(result["Stage 1 WeightedCrossing"] / edgeSum)
    stage2.append(result["Stage 2 WeightedCrossing"] / edgeSum)


# ****************************************************************
# Generating pictures from recoreded results
# ****************************************************************
n = 20
X = stage1
Y = stage2
stage1 = [a for _, a in sorted(zip(Y, X))]
labels = [b for _, b in sorted(zip(Y, range(1, n + 1)))]
Y.sort()
stage2 = Y
# dumbbell
fig, ax = plt.subplots(figsize=(7, 3.8))
ax.set_xticklabels(labels)
ax.set_facecolor("#EFEFEF")
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
plt.vlines(x=range(n), ymin=stage2, ymax=stage1, color="black", alpha=0.4)
plt.scatter(range(n), stage1, color="#7FBFE8", alpha=0.8, label="Markov stage")
plt.scatter(range(n), stage2, color="#6AF588", alpha=1, label="Refinement stage")
handles, labels = ax.get_legend_handles_labels()
ax.legend(reversed(handles), reversed(labels), loc="upper left")
plt.legend()
plt.xticks(range(n))
plt.xlabel("Case ID")
plt.savefig("C:/Users/wasd2/Desktop/sankey_optimizer/example/output/dumbbell")
plt.close()
# box
boxData = [stage1, stage2]
fig, ax = plt.subplots(figsize=(5, 3.8))
ax.set_facecolor("#EFEFEF")
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.set_xticklabels(["Markov stage", "Refinement stage"])
plt.boxplot(boxData)
plt.savefig("C:/Users/wasd2/Desktop/sankey_optimizer/example/output/box")
