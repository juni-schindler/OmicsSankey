import numpy as np
import json
import helper

sumK = 7

caseInfo = {}
np.random.seed(0)
for V in range(9, 13):
    caseInfo[V] = {}
    for n in range(1, sumK):
        totalLineNumber = 0
        for i in range(10):
            linkNumber = helper.getRandCase(V, 2 * n, str(i))
            totalLineNumber += linkNumber
        caseInfo[V][n] = totalLineNumber / 10
    sumK -= 1
with open(
    "C:/Users/wasd2/Desktop/sankey_optimizer/example/input/robust/caseInfo.json", "w"
) as f:
    json.dump(caseInfo, f)
