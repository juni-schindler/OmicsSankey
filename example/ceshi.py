import json
import sys
import os

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
input_dir = sys.argv[1]
with open(input_dir, "r") as f:
    data = json.load(f)
output_dir = sys.argv[2]
alpha1 = 0.01
alpha2 = 0.1
N = 100
M = 100
n = len(data["nodes"])

# ilp
# output_dir = "C:/Users/wasd2/Desktop/sankey_optimizer/example/output/robust_ilp_result.json"

np.random.seed(0)
algo = "BC"
# 运行主函数
result = main.run_main(algo, input_dir, output_dir, n, alpha1, alpha2, N, M)

# 显示结果
print(result)
