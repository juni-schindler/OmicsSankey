import json
import sys

import numpy as np

from sankey_optimizer import main

# 定义输入目录和输出文件
# input_dir = (
#     "/Users/bowentan/projects/sankey_optimizer/example/input/robust/input_9_4/0.json"
# )
# # stage
# output_dir = (
#     "/Users/bowentan/projects/sankey_optimizer/example/output/robust_method_result.json"
# )
min_weighted_crossing = 100000
best_result = None
output_dir = sys.argv[2]
for alpha2 in [0.1, 0.2, 0.3, 0.4, 0.5]:
    input_dir = sys.argv[1]
    with open(input_dir, "r") as f:
        data = json.load(f)
    alpha1 = 0.01
    N = 100
    M = 100
    # n = len(data["nodes"])
    data["level"] = {int(i): v for i, v in data["level"].items()}
    n = len(data["level"].values())
    dummy_signal = False
    cycle_signal = False

    #### For Biosankey data
    # dummy_signal = True
    # cycle_signal = False
    # data["level"] = {int(i): v for i, v in data["level"].items()}
    # n = len(data["level"].values())

    # ilp
    # output_dir = "C:/Users/wasd2/Desktop/sankey_optimizer/example/output/robust_ilp_result.json"

    np.random.seed(0)
    algo = "BC"
    # 运行主函数

    result = main.run_method(
        algo,
        data,
        input_dir,
        n,
        alpha1,
        alpha2,
        N,
        M,
        dummy_signal,
        cycle_signal,
        data["level"],
    )
    print("Stage 1 Crossing:", result["Stage 1 Crossing"])
    print("Stage 1 WeightedCrossing:", result["Stage 1 WeightedCrossing"])
    print("Stage 2 Crossing:", result["Stage 2 Crossing"])
    print("Stage 2 WeightedCrossing:", result["Stage 2 WeightedCrossing"])
    if len(result["Stage 2 Ordering"]) > 0:
        if result["Stage 2 WeightedCrossing"] < min_weighted_crossing:
            min_weighted_crossing = result["Stage 2 WeightedCrossing"]
            best_result = result
        else:
            continue
    else:
        if result["Stage 1 WeightedCrossing"] < min_weighted_crossing:
            min_weighted_crossing = result["Stage 1 WeightedCrossing"]
            best_result = result

# 显示结果
with open(output_dir, "w") as f:
    json.dump(best_result, f)
