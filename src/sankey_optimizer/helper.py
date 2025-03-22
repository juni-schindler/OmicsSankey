import json
import os
from itertools import groupby

import numpy as np
from numpy import linalg as LA

np.set_printoptions(precision=16)


def clearResult(vector):  # 对输入的向量进行排序并返回排序后的索引加 1 的结果。
    index = sorted(range(len(vector)), key=lambda k: vector[k], reverse=True)
    index = np.array(index)
    return np.add(index, np.ones(index.shape))


def getEigen(
    A, i
):  # 计算矩阵 A 的特征值和特征向量，并返回第 i 大特征值对应的特征向量。
    a, b = LA.eig(A)  # [[1.234]]
    indexList = sorted(range(len(a)), key=lambda k: a[k], reverse=True)
    index = indexList[i]  # FIXME:
    vector = b[:, index]
    return vector


def getCrossing(inputMatrix):  # 计算输入矩阵的交叉值。
    npMatrix = np.array(inputMatrix)
    p = npMatrix.shape[0]
    q = npMatrix.shape[1]
    crossing = 0
    for j in range(1, p):
        for k in range(j + 1, p + 1):
            for a in range(1, q):
                for b in range(a + 1, q + 1):
                    crossing += npMatrix[j - 1, b - 1] * npMatrix[k - 1, a - 1]
    return crossing


# --------------------------------------------------------


def calculate_crossings(result, nodes, groupedLinks):
    weightedCrossing = 0
    crossing = 0
    for i in range(0, len(result) - 1):
        order1 = result[i]
        order2 = result[i + 1]
        nodes1 = nodes[i]
        nodes2 = nodes[i + 1]
        m1 = np.empty([len(order1), len(order2)])
        m2 = np.empty([len(order1), len(order2)])
        for j in range(0, len(order1)):
            sourceName = nodes1[int(order1[j]) - 1]["name"]
            for k in range(0, len(order2)):
                targetName = nodes2[int(order2[k]) - 1]["name"]
                value1 = 0
                value2 = 0
                for link in groupedLinks[sourceName]:
                    if (targetName == link["target"]) & (sourceName == link["source"]):
                        value1 = link["value"]
                        value2 = 1
                m1[j, k] = value1
                m2[j, k] = value2
        weightedCrossing += getCrossing(m1)
        crossing += getCrossing(m2)
    return {"weightedCrossing": weightedCrossing, "crossing": crossing}


def load_json(input_dir):
    print(input_dir)
    with open(input_dir, "r") as f:
        data = json.load(f)
    return data


def save_json(output_dir, result):
    with open(output_dir, "w") as outfile:
        json.dump(result, outfile)


# ----------------------------------------------------------


# 将输入矩阵 matrix 和一个随机矩阵按一定权重进行组合
def randomMatrix(matrix, beta):
    orig = np.dot((1 - beta), matrix)
    rand = np.random.rand(matrix.shape[0], matrix.shape[1])  # FIXME:
    for i, arr in enumerate(rand):
        sum = np.sum(rand[i])
        rand[i] = np.dot(beta / sum, arr)
    return np.add(orig, rand)


# 对输入的矩阵列表进行一系列处理，包括随机化矩阵、矩阵相乘、计算特征向量等操作，最终返回处理后的结果和中间向量
def parallel(matrixList, beta=0.1, eigen=1):
    print(matrixList)
    newMatrixList = []
    for i, matrix in enumerate(matrixList):
        matrix = np.array(matrix)
        newMatrixList.append(randomMatrix(matrix, beta))
        if i == 0:
            A = newMatrixList[i]
        else:
            A = np.matmul(A, newMatrixList[i])
    vector = getEigen(A, eigen)
    vectors = []
    result = []
    for i, matrix in enumerate(newMatrixList):
        if i < len(newMatrixList) / 2 + 1:
            if i == 0:
                vectors.append(vector)
                result.append(clearResult(vector))
                vector = np.dot(newMatrixList[len(newMatrixList) - i - 1], vector)
            else:
                vectors.append(vector)
                result.append(clearResult(vector))
                vector = np.dot(newMatrixList[len(newMatrixList) - i - 1], vector)
    return {
        "result": result,
        "vectors": vectors,
    }


# --------------------------------------------------------------run_method
def getNodePos(layer, nodes):
    keys = list(nodes[layer].keys())
    for i in range(len(keys)):
        nodes[layer][keys[i]]["pos"] = (
            len(keys) - nodes[layer][keys[i]]["order"] - 1
        ) / len(keys)


def getAllNodePos(levelNumber, nodes):
    for i in range(levelNumber):
        getNodePos(i, nodes)


def updateLinkOrder(level, orientation, nodes, links):
    priority = ["source", "target"]
    indexes = [level, level + 1]
    if orientation == "right":
        priority.reverse()
        indexes.reverse()
    links[level].sort(
        key=lambda x: (
            nodes[indexes[0]][x[priority[0]]]["order"],
            nodes[indexes[1]][x[priority[1]]]["order"],
        )
    )


def assignLinkPos(level, orientation, nodes, links):
    updateLinkOrder(level, orientation, nodes, links)


def initPos(levelNumber, nodes, links):
    getAllNodePos(levelNumber, nodes)
    for level in range(levelNumber - 1):
        updateLinkOrder(level, "left", nodes, links)
        for i in range(len(links[level])):
            link = links[level][i]
            if (i == 0) | (
                (i != 0) & (links[level][i - 1]["source"] != link["source"])
            ):
                j = nodes[level][link["source"]]["right_edge_number"]
            link["sourcepos"] = nodes[level][link["source"]]["pos"] + (
                j / (nodes[level][link["source"]]["right_edge_number"] + 1)
            ) / len(nodes[level].keys())
            j = j - 1
        updateLinkOrder(level, "right", nodes, links)
        for i in range(len(links[level])):
            link = links[level][i]
            if (i == 0) | (
                (i != 0) & (links[level][i - 1]["target"] != link["target"])
            ):
                j = nodes[level + 1][link["target"]]["left_edge_number"]
            link["targetpos"] = nodes[level + 1][link["target"]]["pos"] + (
                j / (nodes[level + 1][link["target"]]["left_edge_number"] + 1)
            ) / len(nodes[level + 1].keys())
            j = j - 1


def posCalculation(level, leftPoses, rightPoses, nodes, links, alpha2):
    keys = list(nodes[level].keys())
    isLeft = len(leftPoses) == 0
    isRight = len(rightPoses) == 0
    for i in range(len(keys)):
        node = nodes[level][keys[i]]
        # left
        leftPos = 0
        if not isLeft:
            leftRands = np.random.rand(len(links[level - 1]))
            for i in range(len(links[level - 1])):
                leftPos += alpha2 * leftRands[i] * leftPoses[i] / sum(leftRands)
            for l in range(len(node["left_pos"])):
                leftPos += (
                    (1 - alpha2)
                    * node["left_pos"][l]
                    * node["left_weight"][l]
                    / sum(node["left_weight"])
                )
        # right
        rightPos = 0
        if not isRight:
            rightRands = np.random.rand(len(links[level]))
            for i in range(len(links[level])):
                rightPos += alpha2 * rightRands[i] * rightPoses[i] / sum(rightRands)
            for r in range(len(node["right_pos"])):
                rightPos += (
                    (1 - alpha2)
                    * node["right_pos"][r]
                    * node["right_weight"][r]
                    / sum(node["right_weight"])
                )
        if (not isRight) & (not isLeft):
            node["calculatedPos"] = (leftPos + rightPos) / 2
        else:
            node["calculatedPos"] = leftPos + rightPos


def calculateNodePos(
    level, nodes, links, alpha2, dummy_signal, cycle_signal, isLeft=False, isRight=False
):
    keys = list(nodes[level].keys())
    for i in range(len(keys)):
        nodes[level][keys[i]]["left_pos"] = []
        nodes[level][keys[i]]["right_pos"] = []
        nodes[level][keys[i]]["left_weight"] = []
        nodes[level][keys[i]]["right_weight"] = []
    leftPoses = []
    if not isLeft:
        for i in range(len(links[level - 1])):
            link = links[level - 1][i]
            nodes[level][link["target"]]["left_pos"].append(link["sourcepos"])
            if not dummy_signal and not cycle_signal:
                nodes[level][link["target"]]["left_weight"].append(link["value"])
            else:
                nodes[level][link["target"]]["left_weight"].append(
                    np.log10(link["value"] + 1)
                )  # link['value'])
            leftPoses.append(link["sourcepos"])
    rightPoses = []
    if not isRight:
        for i in range(len(links[level])):
            link = links[level][i]
            nodes[level][link["source"]]["right_pos"].append(link["targetpos"])
            if not dummy_signal and not cycle_signal:
                nodes[level][link["source"]]["right_weight"].append(link["value"])
            else:
                nodes[level][link["source"]]["right_weight"].append(
                    np.log10(link["value"] + 1)
                )  # link['value'])
            rightPoses.append(link["targetpos"])
    posCalculation(level, leftPoses, rightPoses, nodes, links, alpha2)


def updateNodeOrder(level, nodes):
    keys = list(nodes[level].keys())
    keys.sort(key=lambda k: nodes[level][k]["calculatedPos"], reverse=True)
    for i in range(len(keys)):
        nodes[level][keys[i]]["order"] = i


def updateLinkPos(level, orientation, nodes, links):
    index = level + 1
    otherOrientation = "source"
    if orientation == "source":
        otherOrientation = "target"
        index = level
    for i in range(len(links[level])):
        link = links[level][i]
        link[orientation + "pos"] = nodes[index][link[orientation]]["pos"] + link[
            otherOrientation + "pos"
        ] / len(nodes[index].keys())


def getOrdering(levelNumber, nodes):
    orders = []
    for i in range(levelNumber):
        keys = list(nodes[i].keys())
        keys.sort(key=lambda k: nodes[i][k]["order"])
        order = []
        for j in range(len(keys)):
            order.append(nodes[i][keys[j]]["id"])
        orders.append(order)
    return orders


# ----------------------------------------------------------------------cycle


def getLevel(i, levels):
    return levels[str(i)]


def getOptimalCase(levels, levelSize, levelNumber, totalSize):
    weights = []
    for i in range(len(levelSize)):
        weight = np.random.multinomial(
            totalSize - levelSize[i], [1 / float(levelSize[i])] * levelSize[i], size=1
        )[0]
        newWeight = []
        for j in range(len(weight)):
            newWeight.append(weight[j] + 1)
        weights.append(list(newWeight))
    # generate nodes
    nodes = []
    level = []
    for i in range(levelNumber):
        node = []
        levelSub = []
        for j in range(levelSize[i]):
            levelSub.append(str(i) + "_" + str(j))
            node.append(
                {
                    "name": str(i) + "_" + str(j),
                    "id": j,
                    "size": weights[i][j],
                    "remainingSizeL": weights[i][j],
                    "remainingSizeR": weights[i][j],
                }
            )
        nodes.append(node)
        level.append(levelSub)
    # genarate non-crossing links
    links = []
    completeLinks = []
    for i in range(levelNumber):
        k = 0
        j = 0
        link = []
        while (not j == levelSize[i]) & (not k == levelSize[getLevel(i + 1, levels)]):
            node1 = nodes[getLevel(i, levels)][j]
            node2 = nodes[getLevel(i + 1, levels)][k]
            if node1["remainingSizeR"] > node2["remainingSizeL"]:
                link.append(
                    {
                        "source": node1["name"],
                        "sourceid": j,
                        "target": node2["name"],
                        "targetid": k,
                        "value": node2["remainingSizeL"],
                    }
                )
                completeLinks.append(
                    {
                        "source": node1["name"],
                        "sourceid": j,
                        "target": node2["name"],
                        "targetid": k,
                        "value": node2["remainingSizeL"],
                    }
                )
                node1["remainingSizeR"] -= node2["remainingSizeL"]
                node2["remainingSizeL"] = 0
                k += 1
            elif node1["remainingSizeR"] == node2["remainingSizeL"]:
                link.append(
                    {
                        "source": node1["name"],
                        "sourceid": j,
                        "target": node2["name"],
                        "targetid": k,
                        "value": node2["remainingSizeL"],
                    }
                )
                completeLinks.append(
                    {
                        "source": node1["name"],
                        "sourceid": j,
                        "target": node2["name"],
                        "targetid": k,
                        "value": node2["remainingSizeL"],
                    }
                )
                node1["remainingSizeR"] = 0
                node2["remainingSizeL"] = 0
                k += 1
                j += 1
            else:
                link.append(
                    {
                        "source": node1["name"],
                        "sourceid": j,
                        "target": node2["name"],
                        "targetid": k,
                        "value": node1["remainingSizeR"],
                    }
                )
                completeLinks.append(
                    {
                        "source": node1["name"],
                        "sourceid": j,
                        "target": node2["name"],
                        "targetid": k,
                        "value": node1["remainingSizeR"],
                    }
                )
                node2["remainingSizeL"] -= node1["remainingSizeR"]
                node1["remainingSizeR"] = 0
                j += 1
        links.append(link)
    addedLinks = completeLinks
    totalWeightedSum = 0
    for i in range(len(links)):
        for j in range(len(links[i]) - 1):
            for k in range(j + 1, len(links[i])):
                link1 = links[i][j]["value"]
                link2 = links[i][k]["value"]
                totalWeightedSum += link1 * link2
    return {
        "links": links,
        "nodes": nodes,
        "completeLinks": completeLinks,
        "edge": totalWeightedSum,
    }


# -------------------------------------------------------robust_weighted_edge_sum


def calculator(input_dir):
    data = load_json(input_dir)
    layeredLinks = data["links"]
    totalWeightedSum = 0
    for i in range(len(layeredLinks)):
        for j in range(len(layeredLinks[i]) - 1):
            for k in range(j + 1, len(layeredLinks[i])):
                link1 = layeredLinks[i][j]["value"]
                link2 = layeredLinks[i][k]["value"]
                totalWeightedSum += link1 * link2
    return totalWeightedSum


# -------------------------------------------------------robust_test_case_generator


def getRandCase(V, n, output):
    if not os.path.exists(
        "C:/Users/wasd2/Desktop/sankey_optimizer/example/input/robust/input"
        + "_"
        + str(V)
        + "_"
        + str(n)
    ):
        os.makedirs(
            os.path.dirname(
                "C:/Users/wasd2/Desktop/sankey_optimizer/example/input/robust/input"
                + "_"
                + str(V)
                + "_"
                + str(n)
                + "/"
                + output
                + ".json"
            )
        )
    levelSize = list(np.full(n, V))
    totalSize = 100
    levelNumber = n
    weights = []
    for i in range(len(levelSize)):
        weight = np.random.multinomial(
            totalSize - levelSize[i], [1 / float(levelSize[i])] * levelSize[i], size=1
        )[0]
        newWeight = []
        for j in range(len(weight)):
            newWeight.append(int(weight[j]) + 1)
        weights.append(list(newWeight))
    nodes = []
    for i in range(levelNumber):
        node = []
        for j in range(levelSize[i]):
            node.append(
                {
                    "name": str(i) + "_" + str(j),
                    "id": j,
                    "size": weights[i][j],
                    "remainingSizeL": weights[i][j],
                    "remainingSizeR": weights[i][j],
                }
            )
        nodes.append(node)

    links = []
    completeLinks = 0
    for i in range(levelNumber - 1):
        link = addLink(i, nodes)
        links.append(link)
        completeLinks += len(link)

    with open(
        "C:/Users/wasd2/Desktop/sankey_optimizer/example/input/robust/input"
        + "_"
        + str(V)
        + "_"
        + str(n)
        + "/"
        + output
        + ".json",
        "w",
    ) as f:
        json.dump({"nodes": nodes, "links": links}, f)

    return completeLinks


def addLink(level, nodes):
    link = []
    left = list(range(len(nodes[level])))
    right = list(range(len(nodes[level + 1])))
    while not (len(left) == 0 & len(right) == 0):
        i = np.random.randint(len(left))
        j = np.random.randint(len(right))
        node1 = nodes[level][left[i]]
        node2 = nodes[level + 1][right[j]]
        if node1["remainingSizeR"] > node2["remainingSizeL"]:
            link.append(
                {
                    "source": node1["name"],
                    "sourceid": left[i],
                    "target": node2["name"],
                    "targetid": right[j],
                    "value": node2["remainingSizeL"],
                }
            )
            node1["remainingSizeR"] -= node2["remainingSizeL"]
            node2["remainingSizeL"] = 0
            right.remove(right[j])
        elif node1["remainingSizeR"] == node2["remainingSizeL"]:
            link.append(
                {
                    "source": node1["name"],
                    "sourceid": left[i],
                    "target": node2["name"],
                    "targetid": right[j],
                    "value": node2["remainingSizeL"],
                }
            )
            node1["remainingSizeR"] = 0
            node2["remainingSizeL"] = 0
            right.remove(right[j])
            left.remove(left[i])
        else:
            link.append(
                {
                    "source": node1["name"],
                    "sourceid": left[i],
                    "target": node2["name"],
                    "targetid": right[j],
                    "value": node1["remainingSizeR"],
                }
            )
            node2["remainingSizeL"] -= node1["remainingSizeR"]
            node1["remainingSizeR"] = 0
            left.remove(left[i])
    return link


# ----------------------------------------------------algorithm
def stage1_data_preprocessing(nodes, layeredLinks, n):
    level = []
    for i in range(len(nodes)):
        l = []
        for j in range(len(nodes[i])):
            l.append(nodes[i][j])
        level.append(l)

    levelNumber = n
    addedLinks = []
    numLink = len(layeredLinks)
    print(layeredLinks)
    for i in range(len(layeredLinks)):
        for j in range(len(layeredLinks[i])):
            print(i, j)
            addedLinks.append(layeredLinks[i][j])

    link1 = addedLinks
    link2 = addedLinks
    link1.sort(key=lambda content: content["source"])
    groups1 = groupby(link1, lambda content: content["source"])
    for source, links in groups1:
        size = 0
        for link in links:
            size += float(link["value"])
        for i in range(n - 1):
            if source in level[i]:
                nodes[i].append(
                    {
                        "name": source,
                        "size": size,
                    }
                )
    link2.sort(key=lambda content: content["target"])
    groups2 = groupby(link2, lambda content: content["target"])
    for target, links in groups2:
        if target in level[levelNumber - 1]:
            size = 0
            for link in links:
                size += float(link["value"])
            nodes[levelNumber - 1].append(
                {
                    "name": target,
                    "size": size,
                }
            )
    addedLinks.sort(key=lambda content: content["source"])
    groups3 = groupby(addedLinks, lambda content: content["source"])
    groupedLinks = {}
    for source, linksss in groups3:
        groupedLinks[source] = list(linksss)
    return {
        "nodes": nodes,
        "layeredLinks": layeredLinks,
        "addedLinks": addedLinks,
        "groupedLinks": groupedLinks,
    }


def dummy_data_preprocessing(layeredLinks, levelNumber, level):
    # add dummy nodes and links
    addedLinks = []
    numLink = len(layeredLinks)
    for index in range(numLink):
        isLong = False
        source = layeredLinks[index]["source"]
        target = layeredLinks[index]["target"]
        for j in range(levelNumber - 1):
            if (source in level[j]) & (target not in level[j + 1]):
                isLong = True
                addedLinks.append(
                    {
                        "source": source,
                        "target": "dummy " + target + str(index) + str(j + 1),
                        "value": layeredLinks[index]["value"],
                    }
                )
                level[j + 1].append("dummy " + target + str(index) + str(j + 1))
                isFound = False
                for l in range(j + 2, levelNumber):
                    if (target not in level[l]) & (not isFound):
                        level[l].append("dummy " + target + str(index) + str(l))
                        addedLinks.append(
                            {
                                "source": "dummy " + target + str(index) + str(l - 1),
                                "target": "dummy " + target + str(index) + str(l),
                                "value": layeredLinks[index]["value"],
                            }
                        )
                    elif target in level[l]:
                        isFound = True
                        addedLinks.append(
                            {
                                "source": "dummy " + target + str(index) + str(l - 1),
                                "target": target,
                                "value": layeredLinks[index]["value"],
                            }
                        )
        if not isLong:
            addedLinks.append(layeredLinks[index])

    # change data structure of links and nodes for better access
    link1 = addedLinks
    link2 = addedLinks
    link1.sort(key=lambda content: content["source"])
    groups1 = groupby(link1, lambda content: content["source"])

    nodes = [[] for _ in range(levelNumber)]

    for source, links in groups1:
        size = 0
        for link in links:
            size += float(link["value"])
        for i in range(levelNumber - 1):
            if source in level[i]:
                nodes[i].append(
                    {
                        "name": source,
                        "size": size,
                    }
                )
    link2.sort(key=lambda content: content["target"])
    groups2 = groupby(link2, lambda content: content["target"])
    for target, links in groups2:
        if target in level[levelNumber - 1]:
            size = 0
            for link in links:
                size += float(link["value"])
            nodes[levelNumber - 1].append(
                {
                    "name": target,
                    "size": size,
                }
            )
    addedLinks.sort(key=lambda content: content["source"])
    groups3 = groupby(addedLinks, lambda content: content["source"])
    groupedLinks = {}
    for source, links in groups3:
        groupedLinks[source] = list(links)

    return {
        "nodes": nodes,
        "layeredLinks": layeredLinks,
        "addedLinks": addedLinks,
        "groupedLinks": groupedLinks,
    }


def cycle_data_preprocessing(nodes, layeredLinks, completeLinks):
    addedLinks = completeLinks
    addedLinks.sort(key=lambda content: content["source"])
    groups3 = groupby(addedLinks, lambda content: content["source"])
    groupedLinks = {}
    for source, linkss in groups3:
        groupedLinks[source] = list(linkss)

    return {
        "nodes": nodes,
        "layeredLinks": layeredLinks,
        "addedLinks": addedLinks,
        "groupedLinks": groupedLinks,
    }


# 为第二阶段的计算准备数据，通过重新组织节点数据并统计每个节点的连接边数量
# 输入：data(和第一段代码有关)、M、nodes(和第一段代码有关)、result(stage1的输出)
# 输出：numLink、links、x、nodes(修改版)、yWeighted和yNonWeighted(未赋值)
# ****************************************************************
# preparing data for Stage 2, not part of the algorithm
# ****************************************************************
def stage2_data_preprocessing(
    nodes,
    layeredLinks,
    result,
    dummy_signal=False,
    cycle_signal=False,
    addedLinks=None,
    levels=None,
):
    links = layeredLinks
    # x = np.arange(M+1)
    # yWeighted = []
    # yNonWeighted = []
    preNodes = nodes
    orders = result
    nodes = []
    for i in range(0, len(preNodes)):
        newLevel = {}
        for j in range(0, len(orders[i])):
            newLevel[preNodes[i][int(orders[i][j]) - 1]["name"]] = {
                "order": j,
                "size": preNodes[i][int(orders[i][j]) - 1]["size"],
                "left_edge_number": 0,
                "right_edge_number": 0,
                "calculatedPos": 0,
                "id": int(orders[i][j]),
            }
        nodes.append(newLevel)

    if dummy_signal:
        links = []
        for j in range(0, len(nodes) - 1):
            linkLevel = []
            i = 0
            for link in addedLinks:
                if link["source"] in nodes[j].keys():
                    link["id"] = i
                    link["value"] = float(link["value"])
                    i += 1
                    linkLevel.append(link)
            links.append(linkLevel)

    for i in range(len(links)):
        for j in range(len(links[i])):
            link = links[i][j]
            if cycle_signal:
                nodes[levels[str(i)]][link["source"]]["right_edge_number"] += 1
                nodes[levels[str(i + 1)]][link["target"]]["left_edge_number"] += 1
            else:
                nodes[i][link["source"]]["right_edge_number"] += 1
                nodes[i + 1][link["target"]]["left_edge_number"] += 1

    return {"nodes": nodes, "layeredLinks": links}
