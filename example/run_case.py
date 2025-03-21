import sankey_optimizer.main

# 定义输入目录和输出文件
input_dir = "/Users/bowentan/projects/sankey_optimizer/example/input/ilp_case.json"
output_file = (
    "/Users/bowentan/projects/sankey_optimizer/example/output/robust_method_result.json"
)
alpha1 = 0.01
alpha2 = 0.1
N = 100
M = 100
n = 4
algo = "BC"
# 运行主函数

result = sankey_optimizer.main.run_main(
    algo, input_dir, output_file, n, alpha1, alpha2, N, M
)

# 显示结果
print(result)
