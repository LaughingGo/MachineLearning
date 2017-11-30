# coding=gbk
# 导入所需要的库
import pandas as pd
import numpy as np
import math
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
import time
import matplotlib.pyplot as plt
plt.style.use('ggplot')

if __name__ == '__main__':
    # 导入数据
    dataSource = pd.read_csv('./Dataset/krkopt.data', sep=',', header=None)
    # 将字符型特征转为数值型变量，由于原始数据的第1,3,5列已经是数值型，所以只需转换第0,2,4列
    le = LabelEncoder()
    le.fit(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
    dataNumerical = dataSource
    dataNumerical.iloc[:][0] = le.transform(dataSource.iloc[:][0])
    dataNumerical.iloc[:][2] = le.transform(dataSource.iloc[:][2])
    dataNumerical.iloc[:][4] = le.transform(dataSource.iloc[:][4])

    # 设置训练数据
    train_data = dataNumerical.iloc[:, 0:6]
    train_target = dataNumerical.iloc[:, 6]
    # 将原始数据打乱（防止过拟合或者不收敛），并按照一定比例划分训练集与测试集。
    X_train, X_test, y_train, y_test = train_test_split(train_data, train_target, test_size=0.2, random_state=0)

    #################### 开始训练 ###################
    # 记录时间
    startTime = time.time()
    # 设置不同数量的基学习器进行迭代训练，统计对应的正确率与训练时间
    estimatorNumbers = np.linspace(2, 40, 20,dtype=int)
    Accuracys = []
    CostTimes = []
    # estimatorNumber = 32
    for estimatorNumber in estimatorNumbers:
        bg_params = {'max_features':6,'oob_score':True,'bootstrap_features': False, 'n_jobs': -1,'n_estimators':estimatorNumber}
        outputs = ['draw', 'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven',
                   'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen']

        # Bag1ToN为存储所有学习器的数组（二维），根据一对多策略，由18个bangging数组组成，每个bagging数组是针对某一特定类的训练分类器
        Bag1ToN=[]
        # 分别针对每一类数据的实际训练轮次
        realStageNumber=np.zeros(18)
        # 训练样本数量
        trainNum = len(X_train)
        # IterativeBagging只适用于回归或二分类问题；
        # 这里将二分类问题扩展成多分类，根据一对多原理，把样本重新划分
        X_train_TwoClass = []
        for tt in range(trainNum):
            X_train_TwoClass.append(X_train.iloc[tt])
        # 针对每一类别训练一个分类器
        for k in range(18):
            # 当前分类类别
            classK= outputs[k]
            # 当前类标记为1，否则为0
            y_train_TwoClasses = [1 if (y_train.iloc[tt]==classK) else 0 for tt in range(len(y_train)) ]
            # 设置最大训练轮次
            stageNumber = 15
            # 残差初始化
            next_Y = y_train_TwoClasses
            # 记录当前类别的分类器集合，由每一轮的分类器组成
            Bag = []
            # 当前最小残差平方和
            minSumofSqu = sum([next_Y[x]*next_Y[x] for x in range(trainNum)])
            realStageNumber[k] = 0
            # 轮次迭代
            for n in range(stageNumber):
                # 当前学习器
                BagTemp = BaggingRegressor(**bg_params)
                BagTemp.fit(X_train_TwoClass,next_Y)

                # 包外估计(保证无偏估计)
                predict_Y = BagTemp.oob_prediction_
                next_Y = [next_Y[m]-predict_Y[m] for m in range(trainNum)]
                Bag.append(BagTemp)
                realStageNumber[k] += 1
                #当前残差（new_Y）平方和大于历史最小值的1.1倍时终止循环
                minSumofSquTemp = sum([next_Y[x] * next_Y[x] for x in range(trainNum)])
                if(minSumofSquTemp>1.1*minSumofSqu):
                    break
                else:
                    if (minSumofSquTemp <minSumofSqu):
                        minSumofSqu = minSumofSquTemp
                print("Classifier#", k+1,"---stage：",n+1)
            Bag1ToN.append(Bag)
            print("Classifier#",k+1,"done!")
        print("Training time:", time.time() - startTime,"(s)")
        # 训练完成，记录时间
        endTime = time.time()
        CostTimes.append(endTime-startTime)
        #################### 预测测试样本分类结果，计算精度###################

        # 初始化变量
        predict_Y_test = np.zeros(len(X_test))
        predict_Y_test_sum = np.zeros([len(X_test), 18])
        max_Index = np.zeros(len(X_test))
        max_sumY = np.zeros(len(X_test))
        # 每个类别对应的分类器都用于测试样本的预测
        for j in range(18):
            Bagj = Bag1ToN[j]
            # 每个残差轮次的预测值累加作为最终的预测值
            for i in range(int(realStageNumber[j])):
                predict_Y_testTemp = Bagj[i].predict(X_test)
                predict_Y_test = [predict_Y_test[j] + predict_Y_testTemp[j] for j in range(len(X_test))]
            for s in range(len(X_test)):
                # 每个样本的回归预测值大于0.5则分为1类，否则分为0类
                if predict_Y_test[s] > 0.5:
                    predict_Y_test_sum[s][j] += 1
                    if (predict_Y_test_sum[s][j] > max_sumY[s]):
                        max_Index[s] = j
                        max_sumY[s] = predict_Y_test_sum[s][j]
                else:
                    for t in range(18):
                        if t != j:
                            predict_Y_test_sum[s][t] += 1
                            if (predict_Y_test_sum[s][t] > max_sumY[s]):
                                max_Index[s] = t
                                max_sumY[s] = predict_Y_test_sum[s][t]
        # 根据18个分类器的预测分类结果，进行投票，最终确定样本所被分为的类别
        predict_Y_test_Final = [outputs[int(max_Index[u])] for u in range(len(y_test))]
        accuracy_test = metrics.accuracy_score(y_test, predict_Y_test_Final)
        Accuracys.append(accuracy_test)

        #################### 预测训练样本分类结果，计算精度###################
        predict_Y_train = np.zeros(len(X_train))
        predict_Y_train_sum = np.zeros([len(X_train), 18])
        max_Index = np.zeros(len(X_train))
        max_sumY = np.zeros(len(X_train))
        for j in range(18):
            Bagj = Bag1ToN[j]
            for i in range(int(realStageNumber[j])):
                predict_Y_trainTemp = Bagj[i].predict(X_train)
                predict_Y_train = [predict_Y_train[j] + predict_Y_trainTemp[j] for j in range(len(X_train))]
            for s in range(len(X_train)):
                if predict_Y_train[s] > 0.5:
                    predict_Y_train_sum[s][j] += 1
                    if (predict_Y_train_sum[s][j] > max_sumY[s]):
                        max_Index[s] = j
                        max_sumY[s] = predict_Y_train_sum[s][j]
                else:
                    for t in range(18):
                        if t != j:
                            predict_Y_train_sum[s][t] += 1
                            if (predict_Y_train_sum[s][t] > max_sumY[s]):
                                max_Index[s] = t
                                max_sumY[s] = predict_Y_train_sum[s][t]

        predict_Y_train_Final = [outputs[int(max_Index[u])] for u in range(len(y_train))]
        accuracy_train = metrics.accuracy_score(y_train, predict_Y_train_Final)

        print("accuracy_test:", accuracy_test)
        print("accuracy2:-train", accuracy_train)
        print("Estimator",estimatorNumber,"done!")

    # 绘制基分类器数量——分类精度关系图
    plt.figure('分类精度')
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.title("IterativeBagging基分类器数量——分类精度关系图")
    plt.xlabel("基分类器数量")
    plt.ylabel("分类精度")
    plt.plot(estimatorNumbers, Accuracys, label='IterativeBagging')
    plt.legend(loc='lower right')

    # 绘制基分类器数量——运行时间关系图
    plt.figure('训练时间')
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.title("IterativeBagging基分类器数量——运行时间关系图")
    plt.plot(estimatorNumbers, CostTimes, label='IterativeBagging')
    plt.legend(loc='lower right')
    plt.xlabel("基分类器数量")
    plt.ylabel("运行时间")
    plt.show()

    # 打印结果
    print("Accuracys:", Accuracys)
    print("Costtime:", CostTimes)
