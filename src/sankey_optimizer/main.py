from __future__ import division

from pulp import LpMinimize, LpProblem

from . import helper
from .algorithm import BCAlgorithm, ILPAlgorithm


def run_method(
    algo,
    data,
    input_dir,
    n=None,
    alpha1=None,
    alpha2=None,
    N=None,
    M=None,
    dummy_signal=False,
    cycle_signal=False,
    level=None,
):
    # 当cycle_signal==True时，nodes=None
    # 当algo=='BC'时，input_dir=None
    # 当采用模拟数据时，最后三个参数均为None
    if algo == "BC":
        # 实例化 Sankey 算法
        sankey_algo = BCAlgorithm()

        # 预处理数据
        if dummy_signal:
            stage1_data_pre = helper.dummy_data_preprocessing(data["links"], n, level)
        elif cycle_signal:
            stage1_data_pre = helper.cycle_data_preprocessing(
                data["nodes"], data["links"], data["completeLinks"]
            )
        else:
            stage1_data_pre = helper.stage1_data_preprocessing(
                data["nodes"], data["links"], n
            )

        # 阶段 1：初始节点排序
        result_1 = sankey_algo.stage_1(
            stage1_data_pre["nodes"],
            stage1_data_pre["addedLinks"],
            stage1_data_pre["groupedLinks"],
            n,
            alpha1,
            N,
            dummy_signal,
            cycle_signal,
            level,
        )
        float_ordering = result_1["result"]
        stage1_ordering = [[int(num) for num in sublist] for sublist in float_ordering]

        # 计算阶段 1 的交叉值
        stage1_crossings = helper.calculate_crossings(
            result_1["result"], result_1["nodes"], result_1["groupedLinks"]
        )

        # 预处理数据
        stage2_data_pre = helper.stage2_data_preprocessing(
            result_1["nodes"],
            stage1_data_pre["layeredLinks"],
            result_1["result"],
            dummy_signal,
            cycle_signal,
            result_1["addedLinks"],
            level,
        )

        # 阶段 2：细化节点排序
        result_2 = sankey_algo.stage_2(
            stage2_data_pre["nodes"],
            stage2_data_pre["layeredLinks"],
            result_1["levelNumber"],
            result_1["stage1Result"],
            result_1["groupedLinks"],
            result_1["nodes"],
            n,
            alpha2,
            M,
            dummy_signal,
            cycle_signal,
        )
        stage2_ordering = result_2["result"]

        # 计算阶段 2 的交叉值
        stage2_crossings = helper.calculate_crossings(
            stage2_ordering, result_1["nodes"], result_2["groupedLinks"]
        )

        # 返回结果
        return {
            "Stage 1 Ordering": stage1_ordering,
            "Stage 1 WeightedCrossing": stage1_crossings["weightedCrossing"],
            "Stage 1 Crossing": stage1_crossings["crossing"],
            "Stage 2 Ordering": stage2_ordering,
            "Stage 2 WeightedCrossing": stage2_crossings["weightedCrossing"],
            "Stage 2 Crossing": stage2_crossings["crossing"],
            "minAchievedIteration": result_2["minAchievedIteration"],
        }

    elif algo == "ILP":
        # 实例化 ILP 算法
        ilp_algo = ILPAlgorithm()
        prob = LpProblem(input_dir, LpMinimize)
        result = ilp_algo.ilp(prob, data["nodes"], data["links"])
        return {"value": result["value"], "status": result["status"]}

    else:
        raise ValueError(f"未知算法: {algo}")


def run_main(
    algo,
    input_dir,
    output_dir,
    n=None,
    alpha1=None,
    alpha2=None,
    N=None,
    M=None,
    dummy_signal=False,
    cycle_signal=False,
    level=None,
):
    data = helper.load_json(input_dir)

    # layeredLinks = data['links']
    # nodes = data['nodes']

    result = {}

    result = run_method(
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
        level,
    )

    helper.save_json(output_dir, result)
    return result
