import random
from numpy import var, nonzero, mean, median
import numpy as np


# 对连续标签的处理：计算平方误差 = 方差 * 样本数
def cal_error(dataset):
    v = var(dataset[:, -1])
    error = v * len(dataset)
    return error

# 按照特征号以及特征阈值对数据集进行分离操作
def split_dataset(dataset, feature_id, value):
    # 将相关特征 < value 的分为一类，作为左子树的组成部分
    left = dataset[nonzero(dataset[:, feature_id] < value)]
    # 将相关特征 >= value 的分为一类，作为左子树的组成部分
    right = dataset[nonzero(dataset[:, feature_id] >= value)]
    return left, right


# 选取误差最小的特征来分离数据集
def choose_feature_to_split(dataset):
    
    best_feature_id = 0
    best_value = 0
    lowest_error = 1000000

    # 特征数m
    m = len(dataset[0]) - 1

    # 遍历各个特征
    for feature_id in range(m):
        value = median(dataset[:, feature_id])
        left, right = split_dataset(dataset, feature_id, value)

        # 定义 当前划分方式的误差 = 左子树的误差 + 右子树的误差
        cur_error = cal_error(left) + cal_error(right)
        if cur_error < lowest_error:
            lowest_error = cur_error
            best_feature_id = feature_id
            best_value = value
    total_error = cal_error(dataset)
    # 如果整体误差与当前划分误差较接近，可以不用继续划分
    if total_error < lowest_error:
        return None, None
    return best_feature_id, best_value


# 建立回归树
def create_regression_tree(dataset):
    feature_id, value = choose_feature_to_split(dataset)

    # 没有找到好的特征划分点时，直接取当前标签的均值
    if feature_id == None:
        return mean(dataset[:, -1])

    # 否则以字典的形式定义当前结点
    else:
        node = {}
        node['feature_id'] = feature_id
        node['value'] = value
        left, right = split_dataset(dataset, feature_id, value)

        # 继续划分左右子树
        node['left'] = create_regression_tree(left)
        node['right'] = create_regression_tree(right)

        return node


# 预测
def predict(node, data):
    # 到达叶结点，返回预测结果
    if not isinstance(node, dict):
        return node
    else:
        # 根据当前特征的阈值来判断进入左/右子树
        feature_id = node['feature_id']
        value = node['value']
        if data[feature_id] < value:
            return predict(node['left'], data)
        else:
            return predict(node['right'], data)


# 采用MAPE来对回归树模型做评估
# MAPE 平均绝对百分比误差
def evaluate(tree, dataset):
    err = 0.0
    for data in dataset:
        # 计算预测值
        pre = predict(tree, data)
        # 计算当前绝对百分比误差
        err += pow(abs(data[-1]-pre) / data[-1], 2)
    return err / len(dataset)


# 按照s:1-s的比例划分数据集
def load_booston(p):
    f = open('data/boston_housing.csv', 'r')

    content = f.read().split('\n')

    feature_names = content[0].split(',')[0:-1]

    content = content[1:-1]
    # random.shuffle(content)

    n = len(content)
    m = len(feature_names)

    dataset = np.zeros((n, m+1))

    for i in range(n):
        line = content[i].split(',')
        dataset[i] = line
    s = (int)(p * n)
    return dataset[0:s], dataset[s:-1]


def load_student(p=0.8):
    f = open('data/student.csv', 'r')

    content = f.read().split('\n')

    feature_names = content[0].split(',')[1:-1]

    content = content[1:-1]
    # random.shuffle(content)

    n = len(content)
    m = len(feature_names)

    # random.shuffle(content)

    dataset = np.zeros((n, m + 1))

    for i in range(n):
        line = content[i].split(',')
        line.pop(0)
        # line[-1] = f[line[-1]]
        dataset[i] = line

    s = (int)(p * n)

    return dataset[0:s], dataset[s:-1]


