import cart_regression as rt
import cart_classification as clt

if __name__ == '__main__':

    # 回归模型测试

    boston_train_dataset, boston_test_dataset = rt.load_booston(0.8)
    boston_tree = rt.create_regression_tree(boston_train_dataset)
    boston_MAPE = rt.evaluate(boston_tree, boston_test_dataset)
    print("波士顿房价数据，测试集平均绝对百分比误差 ：", boston_MAPE*100, "%")

    student_train_dataset, student_test_dataset = rt.load_student(0.8)
    student_tree = rt.create_regression_tree(student_train_dataset)
    student_MAPE = rt.evaluate(student_tree, student_test_dataset)
    print("学生录取数据，测试集平均绝对百分比误差 ：", student_MAPE * 100, "%")

    print("回归模型测试结束------------------------------------")

    print("\n")


    # 分类模型测试

    Iris_train_dataset, Iris_test_dataset = clt.load_Iris()
    Iris_tree = clt.create_classification_tree(Iris_train_dataset)
    Iris_accuracy = clt.evaluate(Iris_tree, Iris_test_dataset)
    print("鸢尾花数据，测试准确度 ：", Iris_accuracy * 100, "%")

    zoo_train_dataset, zoo_test_dataset = clt.load_zoo()
    zoo_tree = clt.create_classification_tree(zoo_train_dataset)
    zoo_accuracy = clt.evaluate(zoo_tree, zoo_test_dataset)
    print("动物园数据，测试准确度 ：", zoo_accuracy * 100, "%")
    print("分类模型测试结束-----------------------------------")