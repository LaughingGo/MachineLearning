# -*- coding: utf - 8 - *-
# 导入需要的库
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
import time
import matplotlib.pyplot as plt
plt.style.use('ggplot')


if __name__ == '__main__':
    # 导入数据
    dataSource = pd.read_csv('./Dataset/krkopt.data', sep=',', header=None)

    # 将字符型特征转为数值型变量，由于原始数据的第1,3,5列已经是数值型，所以只需转换第0,2,4列
    le = LabelEncoder()
    le.fit(['a','b','c','d','e','f','g','h'])
    dataNumerical = dataSource
    dataNumerical.iloc[:][0] = le.transform(dataSource.iloc[:][0])
    dataNumerical.iloc[:][2] = le.transform(dataSource.iloc[:][2])
    dataNumerical.iloc[:][4] = le.transform(dataSource.iloc[:][4])

    # 设置训练数据
    train_data = dataNumerical.iloc[:, 0:6]
    train_target = dataNumerical.iloc[:,6]
    # 将原始数据打乱（防止过拟合或者不收敛），并按照一定比例划分训练集与测试集。
    X_train, X_test, y_train, y_test =   train_test_split(train_data, train_target, test_size=0.2, random_state=0)
    # 设置不同数量的基学习器进行迭代训练，统计对应的正确率与训练时间
    estimators = np.linspace(1, 100, 100,dtype=int)

    ############## Bagging 训练 ################

    # 记录每个迭代次数下的训练时间
    Bag_times = []
    # 记录每个迭代次数下分类精度
    Bag_accs = []
    # 参数预设
    bg_params = {'bootstrap_features': True, 'n_jobs': -1}
    # 对每个迭代次数进行迭代训练
    for i in estimators:
        if i == 0:
            x = 1
        else:
            x = i
        # 开始训练，记录时间
        starttime = time.time()
        bg_params['n_estimators'] = x
        Bag = BaggingClassifier(**bg_params)
        Bag.fit(X_train, y_train)
        Bag_time = (time.time() - starttime)

        Bag_times.append(Bag_time)
        # 计算分类精度
        y_predClass = Bag.predict(X_test)
        Bag_acc = metrics.accuracy_score(y_test, y_predClass)
        Bag_accs.append(Bag_acc)
        print("Bagging Estimator",x,"done!")


    ############## RandomForest 训练 ################
    # 参数预设
    params = {'max_features': 'sqrt', 'max_depth': 50, 'min_samples_split': 2,
              'min_samples_leaf': 1,'n_jobs':-1}
    # 记录在每个学习器数量下对应的训练时间
    RF_times = []
    # 记录在每个学习器数量下对应的分类精度
    RF_accs = []
    # 对每个学习器数量设置不同值进行迭代训练
    for i in estimators:
        if i==0:
            x=1
        else:
            x=i
        # 开始训练，记录时间
        starttime = time.time()
        params['n_estimators'] = x
        RF = RandomForestClassifier(**params)
        RF.fit(X_train, y_train)
        RF_time = (time.time() - starttime)
        RF_times.append(RF_time)

        # 计算分类精度
        y_predClass = RF.predict(X_test)
        RF_acc = metrics.accuracy_score(y_test, y_predClass)
        RF_accs.append(RF_acc)
        print("RandomForest Estimator", x, "done!")


    # 绘制训练时间比较图
    plt.figure('训练时间比较')
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.title("RandomForest与Bagging训练时间比较")
    plt.plot(estimators, RF_times,label='RandomForest')
    plt.plot(estimators, Bag_times, label='Bagging')
    plt.legend(loc='lower right')
    plt.xlabel("基分类器数量")
    plt.ylabel("运行时间")

    # 绘制精度比较图
    plt.figure('分类精度比较')
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.title("RandomForest与Bagging分类精度比较")
    plt.xlabel("基分类器数量")
    plt.ylabel("分类精度")
    plt.plot(estimators, RF_accs,label='RandomForest')
    plt.plot(estimators, Bag_accs,label='Bagging')
    plt.legend(loc='lower right')
    # 显示绘图
    plt.show()

    # 打印结果
    print("RandomForest:")
    print("     Cost time:", RF_times)
    print("     Accuracy rate:", RF_accs)
    print("Bagging:")
    print("     Cost time:", Bag_times)
    print("     Accuracy rate:", Bag_accs)
