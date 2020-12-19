from collections import Counter
from numpy import var, nonzero, median
import numpy as np


# 基尼指数计算
def cal_gini(dataset):
    n = len(dataset)
    label_dict = Counter(dataset[:, -1])
    gini = 1.0
    for label_name, label_count in label_dict.items():
        p = label_count / n
        gini -= p * p
    return gini


# 按照特征号以及特征阈值对数据集进行分离操作
def split_dataset(dataset, feature_id, value):
    # 将相关特征 < value 的分为一类，作为左子树的组成部分
    left = dataset[nonzero(dataset[:, feature_id] < value)]
    # 将相关特征 >= value 的分为一类，作为左子树的组成部分
    right = dataset[nonzero(dataset[:, feature_id] >= value)]
    return left, right


# 选取基尼指数最小的特征来分离数据集
def choose_feature_to_split(dataset):

    # 如果当前数据集只有一种标签，直接返回
    if len(set(dataset[:, -1])) == 1:
        return None, None

    best_feature_id = 0
    best_value = 0
    lowest_error = 1000000

    # 特征数m
    m = len(dataset[0]) - 1

    # 遍历各个特征
    for feature_id in range(m):
        value = median(dataset[:, feature_id])
        left, right = split_dataset(dataset, feature_id, value)

        # 定义 当前划分方式的基尼指数 = 左子树基尼指数 * 左子树占比 + 右子树基尼指数 * 右子树占比
        cur_error = cal_gini(left) * len(left) / len(dataset) + cal_gini(right) * len(right) / len(dataset)
        if cur_error < lowest_error:
            lowest_error = cur_error
            best_feature_id = feature_id
            best_value = value
    return best_feature_id, best_value


# 建立分类树
def create_classification_tree(dataset):
    feature_id, value = choose_feature_to_split(dataset)

    # 没有找到好的特征划分点时，直接取当前出现次数最多的标签
    if feature_id == None:
        return Counter(dataset[:, -1]).most_common(1)[0][0]

    # 否则以字典的形式定义当前结点
    else:
        node = {}
        node['feature_id'] = feature_id
        node['value'] = value
        left, right = split_dataset(dataset, feature_id, value)

        # 继续划分左右子树
        node['left'] = create_classification_tree(left)
        node['right'] = create_classification_tree(right)

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


# 采用预测准确度来对回归树模型做评估
def evaluate(tree, dataset):
    corr = 0
    for data in dataset:
        # 计算预测分类值
        pre = predict(tree, data)
        corr += (pre == data[-1])
    return corr / len(dataset)


def load_Iris(p=0.8):
    f = open('data/Iris.csv', 'r')

    content = f.read().split('\n')

    feature_names = content[0].split(',')[1:-1]

    content = content[1:-1]
    # random.shuffle(content)

    n = len(content)
    m = len(feature_names)

    # random.shuffle(content)

    dataset = np.zeros((n, m + 1))

    f = {
        'Iris-setosa': 1,
        'Iris-versicolor': 2,
        'Iris-virginica': 3
    }

    for i in range(n):
        line = content[i].split(',')
        line.pop(0)
        line[-1] = f[line[-1]]
        dataset[i] = line

    s = (int)(p * n)

    return dataset[0:s], dataset[s:-1]

def load_zoo(p=0.8):
    f = open('data/zoo.csv', 'r')

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


# train_dataset, test_dataset = load_zoo()
# tree = create_classification_tree(train_dataset)
# print(tree)
# print(evaluate(tree, test_dataset))


