import numpy as np
import json
import helper

result = {}
sumK = 6
for V in range(9, 13):
    result[V] = {}
    for n in range(1, sumK):
        result[V][n] = []
        for i in range(10):
            input_dir = (
                "C:/Users/wasd2/Desktop/sankey_optimizer/example/input/robust/input"
                + "_"
                + str(V)
                + "_"
                + str(2 * n)
                + "/"
                + str(i)
                + ".json"
            )
            r = helper.calculator(input_dir)
            result[V][n].append(r)
    sumK -= 1

with open(
    "C:/Users/wasd2/Desktop/sankey_optimizer/example/output/robust_weighted_edge_sum.json",
    "w",
) as outfile:
    json.dump(result, outfile)
