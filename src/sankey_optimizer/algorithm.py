from itertools import groupby

import numpy as np
from pulp import LpInteger, LpStatus, LpVariable, value

from . import helper

np.set_printoptions(suppress=True)


class BCAlgorithm:
    def stage_1(
        self,
        nodes,
        addedLinks,
        groupedLinks,
        n,
        alpha1,
        N,
        dummy_signal=False,
        cycle_signal=False,
        level=None,
    ):
        # 根据节点和链接数据，分别从正向和反向构建转移概率矩阵，并将这些矩阵存储在 matrices 列表中
        # 输入: n、nodes(和上段代码有关)、groupedLinks(和上段代码有关)、addedLinks(和上段代码有关) ；
        # 输出：matrices(正反向矩阵)、groupedLinks(重新分组)
        # ****************************************************************
        # Generating matrices for Stage 1
        # ****************************************************************
        levelNumber = n
        matrices = []
        for i in range(n - 1):
            # print(i)
            matrix = []
            for node in nodes[i]:
                # print(node)
                row = []
                # cycle
                if cycle_signal:
                    for j in range(len(nodes[helper.getLevel(i + 1, level)])):
                        preNode = nodes[helper.getLevel(i + 1, level)][j]
                        value = 0
                        for link in groupedLinks[node["name"]]:
                            if preNode["name"] == link["target"]:
                                value = np.log10(float(link["value"]) + 1)
                                # value = float(link['value'])
                        row.append(value)
                    matrix.append([float(j) / sum(row) for j in row])
                # dummy和normal
                else:
                    for preNode in nodes[i + 1]:
                        value = 0
                        for link in groupedLinks[node["name"]]:
                            if preNode["name"] == link["target"]:
                                value = float(link["value"])
                                if dummy_signal:
                                    value = np.log10(float(link["value"]) + 1)
                        row.append(value)
                    matrix.append([float(j) / sum(row) for j in row])  # FIXME:

            matrix = np.array(matrix)
            matrices.append(matrix)

        addedLinks.sort(key=lambda content: content["target"])
        groups3 = groupby(addedLinks, lambda content: content["target"])

        groupedLinks = {}
        for target, links in groups3:
            groupedLinks[target] = list(links)

        for i in range(n - 1, 0, -1):
            matrix = []
            # cycle
            if cycle_signal:
                for node in nodes[helper.getLevel(i, level)]:
                    row = []
                    for preNode in nodes[i - 1]:
                        value = 0
                        for link in groupedLinks[node["name"]]:
                            if preNode["name"] == link["source"]:
                                # value = float(link['value'])
                                value = np.log10(float(link["value"]) + 1)
                        row.append(value)
                    matrix.append([float(j) / sum(row) for j in row])
            # dummy和normal
            else:
                for node in nodes[i]:
                    row = []
                    for preNode in nodes[i - 1]:
                        value = 0
                        for link in groupedLinks[node["name"]]:
                            if preNode["name"] == link["source"]:
                                value = float(link["value"])
                                if dummy_signal:
                                    value = np.log10(float(link["value"]) + 1)
                        row.append(value)
                    matrix.append([float(j) / sum(row) for j in row])

            matrix = np.array(matrix)
            matrices.append(matrix)

        addedLinks.sort(key=lambda content: content["source"])
        groups3 = groupby(addedLinks, lambda content: content["source"])
        groupedLinks = {}
        for source, linksss in groups3:
            groupedLinks[source] = list(linksss)

        # 计算每次计算结果的交叉值，然后选择加权交叉值最小的结果作为第一阶段的优化结果
        # 输入：N、matrices(和上段代码有关)、alpha1、nodes(和第一段代码有关)、groupedLinks（和上段有关)；
        # 输出：stage1Result、result、toPrintResult
        # ****************************************************************
        # stage 1
        # ****************************************************************
        resultArray = [0] * N
        for index in range(0, N):
            resultObj = helper.parallel(matrices, alpha1)
            result = resultObj["result"]

            if cycle_signal:
                # add the crossing between the last and the first layer
                result0 = result[0]
                result.append(result0)

            stage1_cross = helper.calculate_crossings(result, nodes, groupedLinks)

            resultArray[index] = {
                "weightedCrossing": stage1_cross["weightedCrossing"],
                "crossing": stage1_cross["crossing"],
                "order": result,
            }
        resultArray.sort(key=lambda x: x["weightedCrossing"], reverse=False)
        stage1Result = resultArray[0]["weightedCrossing"]
        result = resultArray[0]["order"]
        toPrintResult = []
        for r in range(len(result)):
            toPrintResult.append(list(result[r]))
        return {
            "nodes": nodes,
            "levelNumber": levelNumber,
            "stage1Result": stage1Result,
            "groupedLinks": groupedLinks,
            "result": toPrintResult,
            "addedLinks": addedLinks,
        }

    # -------------------------------------------------------------------------------------------
    # 为第二阶段的计算准备数据，通过重新组织节点数据并统计每个节点的连接边数量
    # 输入：data(和第一段代码有关)、M、nodes(和第一段代码有关)、result(stage1的输出)
    # 输出：numLink、links、x、nodes(修改版)、yWeighted和yNonWeighted(未赋值)
    # ****************************************************************
    # preparing data for Stage 2, not part of the algorithm
    # ****************************************************************
    def stage_2(
        self,
        nodes,
        links,
        levelNumber,
        stage1Result,
        groupedLinks,
        preNodes,
        n,
        alpha2,
        M,
        dummy_signal=False,
        cycle_signal=False,
    ):
        # 在第一阶段结果的基础上进一步优化节点的排序，以最小化加权交叉值
        # 输入：n、M、data(和第一段代码有关)、stage1Result(stage1的输出)、nodes(stage1的修改版)、links(和上两段代码有关)、groupedLinks(和stage1代码有关)
        # 输出：dict(包含阶段 1 和阶段 2 的最小加权交叉值)
        # ****************************************************************
        # Stage 2
        # ****************************************************************
        helper.initPos(levelNumber, nodes, links)
        minWeightedCrossing = stage1Result
        correspondingCrossing = 0
        correspondingOrdering = []
        minAchievedIteration = 0
        for index in range(M):
            for i in range(1, n - 1):
                helper.calculateNodePos(
                    i, nodes, links, alpha2, dummy_signal, cycle_signal
                )
                helper.updateNodeOrder(i, nodes)
                helper.getNodePos(i, nodes)
                helper.updateLinkPos(i - 1, "target", nodes, links)
                helper.updateLinkPos(i, "source", nodes, links)

            for i in [n - 1]:
                helper.calculateNodePos(
                    i, nodes, links, alpha2, dummy_signal, cycle_signal, isRight=True
                )
                helper.updateNodeOrder(i, nodes)
                helper.getNodePos(i, nodes)
                helper.updateLinkPos(i - 1, "target", nodes, links)

            for i in range(n - 2, 0, -1):
                helper.calculateNodePos(
                    i, nodes, links, alpha2, dummy_signal, cycle_signal
                )
                helper.updateNodeOrder(i, nodes)
                helper.getNodePos(i, nodes)
                helper.updateLinkPos(i - 1, "target", nodes, links)
                helper.updateLinkPos(i, "source", nodes, links)

            for i in [0]:
                helper.calculateNodePos(
                    i, nodes, links, alpha2, dummy_signal, cycle_signal, isLeft=True
                )
                helper.updateNodeOrder(i, nodes)
                helper.getNodePos(i, nodes)
                helper.updateLinkPos(i, "source", nodes, links)

            result = helper.getOrdering(levelNumber, nodes)

            checknodes = preNodes
            stage2_cross = helper.calculate_crossings(result, checknodes, groupedLinks)
            weightedCrossing = stage2_cross["weightedCrossing"]
            crossing = stage2_cross["crossing"]

            # print(weightedCrossing, correspondingOrdering)
            if weightedCrossing < minWeightedCrossing:
                minWeightedCrossing = weightedCrossing
                correspondingCrossing = crossing
                correspondingOrdering = result
                minAchievedIteration = index

        # return result for this case
        # print(correspondingOrdering)
        return {
            "nodes": nodes,
            "groupedLinks": groupedLinks,
            "result": correspondingOrdering,
            "minWeightedCrossing": minWeightedCrossing,
            "minAchievedIteration": minAchievedIteration,
        }


class ILPAlgorithm:
    def ilp(self, prob, nodes, links):
        # 定义线性规划问题的变量
        x = []
        for i in range(len(nodes)):
            x_i = []
            for j in range(len(nodes[i])):
                x_i_j = []
                for k in range(len(nodes[i])):
                    if j != k:
                        x_jk = LpVariable(
                            nodes[i][j]["name"] + nodes[i][k]["name"], 0, 1, LpInteger
                        )
                        x_i_j.append(x_jk)
                    else:
                        x_i_j.append(0)
                x_i.append(x_i_j)
            x.append(x_i)

        c = []
        for i in range(len(links)):
            c_i = []
            for j in range(len(links[i])):
                c_i_j = []
                for k in range(len(links[i])):
                    source1 = links[i][j]["sourceid"]
                    target1 = links[i][j]["targetid"]
                    source2 = links[i][k]["sourceid"]
                    target2 = links[i][k]["targetid"]
                    if (j != k) & (source1 != source2) & (target1 != target2):
                        c_jk = LpVariable(
                            str(i)
                            + "_"
                            + str(links[i][j]["sourceid"])
                            + "_"
                            + str(links[i][j]["targetid"])
                            + "_"
                            + str(links[i][k]["sourceid"])
                            + "_"
                            + str(links[i][k]["targetid"]),
                            0,
                            1,
                            LpInteger,
                        )
                        c_i_j.append(
                            {
                                "var": c_jk,
                                "source1": links[i][j]["sourceid"],
                                "target1": links[i][j]["targetid"],
                                "source2": links[i][k]["sourceid"],
                                "target2": links[i][k]["targetid"],
                                "weight": links[i][k]["value"] * links[i][j]["value"],
                            }
                        )
                    else:
                        c_i_j.append(0)
                c_i.append(c_i_j)
            c.append(c_i)

        # 定义目标函数
        # obj
        obj = 0
        for i in range(len(c)):
            for j in range(len(c[i])):
                for k in range(len(c[i][j])):
                    if c[i][j][k] != 0:
                        obj += c[i][j][k]["weight"] * c[i][j][k]["var"]
        prob += obj, "obj"

        # 定义约束条件
        # cond 1
        for i in range(len(x)):
            for j in range(len(x[i])):
                for k in range(j + 1, len(x[i])):
                    prob += x[i][j][k] + x[i][k][j] == 1

        # cond 2
        for i in range(len(x)):
            for a in range(len(x[i])):
                for b in range(a + 1, len(x[i])):
                    for d in range(b + 1, len(x[i])):
                        prob += x[i][a][d] >= x[i][a][b] + x[i][b][d] - 1

        # cond 3
        for i in range(len(c)):
            for j in range(len(c[i])):
                for k in range(len(c[i][j])):
                    if j != k:
                        cross = c[i][j][k]
                        if cross != 0:
                            prob += (
                                cross["var"]
                                + x[i][cross["source2"]][cross["source1"]]
                                + x[i + 1][cross["target1"]][cross["target2"]]
                                >= 1
                            )
                            prob += (
                                cross["var"]
                                + x[i][cross["source1"]][cross["source2"]]
                                + x[i + 1][cross["target2"]][cross["target1"]]
                                >= 1
                            )

        # additional cond 1
        for i in range(len(c)):
            for j in range(len(c[i])):
                for k in range(j + 1, len(c[i])):
                    if c[i][j][k] != 0:
                        prob += c[i][j][k]["var"] == c[i][k][j]["var"]

        # 求解
        status = prob.solve()
        return {"value": value(prob.objective) / 2, "status": LpStatus[prob.status]}
