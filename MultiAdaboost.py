# -*- coding: utf - 8 - *-
# 导入需要的库
import pandas as pd
import numpy as np
import math
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import time
import matplotlib.pyplot as plt
plt.style.use('ggplot')

# 定义二分查找函数，用于根据所产生的随机数来计算对应的样本索引，从而产生服从相应权值概率分布的样本集
def FindIndex(value, array, startIndex, endIndex):
    length = endIndex - startIndex + 1
    if (value <= array[0]):
        index = 0
    else:
        if (startIndex + 1 == endIndex):
            index = endIndex
        else:
            newIndex = startIndex + math.floor(length / 2)

            if (value > array[newIndex]):
                index = FindIndex(value, array, newIndex, endIndex)
            else:
                if (value < array[newIndex]):
                    index = FindIndex(value, array, startIndex, newIndex)
                else:
                    index = newIndex
    return index


if __name__ == '__main__':
    # 读取数据
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
    X_train, X_test, y_train, y_test = train_test_split(train_data, train_target, test_size=0.3,
                                                                         random_state=0)
    # 最大迭代次数，在训练过程中记录不同数量的基学习器进行迭代训练，统计对应的正确率与训练时间
    T = 50
    Accuracys = []
    CostTimes = []
    #T = 25
    # 计算训练轮次，可理解为所训练的Adaboost基学习器个数
    n = math.floor(math.sqrt(T))
    # 计算论文所述的每一轮次对应的I值
    I = [math.ceil(i * T / n) for i in range(1, n)]
    I.append(T)

    # 开始训练，记录时间
    startTime = time.time()
    # 记录轮次，初始为0
    k = 0
    # 样本空间大小
    sampleNumber = len(X_train)
    # 初始化样本权值为均匀分布
    weightStd = [1 / sampleNumber for q in range(sampleNumber)]
    # 根据样本权值，记录每个样本在数轴[0,1]范围内所对应的区间，以方便根据随机数来产生对应样本
    weightStdsum = [sum(weightStd[:q]) for q in range(1, sampleNumber)]
    weightStdsum.append(1)
    # 错误率
    epsilon = []
    # 分类器评估指标beta
    beta = []
    # 分类器集合C
    C = []

    predict_time = 0
    # 开始迭代
    for i in range(1, T + 1):
        if (k < n):
            Ik = I[k]
        else:
            Ik = T
        # 判断如果迭代次数 i 到达当前轮次训练次数上限，则终止本轮训练
        if (Ik == i):
            # 初始化样本权值为为泊松分布，并进行标准化到1
            weight = [-math.log(random.random() * 999 / 1000) for j in range(sampleNumber)]
            sumWeight = sum(weight)
            weightStd = [weight[p] / sumWeight for p in range(sampleNumber)]
            # 根据样本权值，记录每个样本在数轴[0,1]范围内所对应的区间，以方便根据随机数来产生对应样本
            weightStdsum = [sum(weightStd[:q]) for q in range(1, sampleNumber)]
            weightStdsum.append(1)
            # 轮次加1
            k = k + 1

        ### 按样本权重生成样本集合 ###
        newInputs_X = []
        newInputs_y = []
        for x in range(sampleNumber):
            # 根据所产生随机数落在[0,1]范围内的区间位置，利用二分查找，找到对应的样本索引，从而组成样本空间
            value = random.random()
            index = FindIndex(value, weightStdsum, 0, sampleNumber - 1)
            newInputs_X.append(X_train.iloc[index][:])
            newInputs_y.append(y_train.iloc[index])

        # 选取决策数分类器作为基学习器
        Dtc = DecisionTreeClassifier()
        Dtc.fit(newInputs_X, newInputs_y)
        # 计算当前分类器的分类错误率
        y_predict = Dtc.predict(newInputs_X)
        error = [weightStd[s] for s in range(sampleNumber) if y_predict[s] != newInputs_y[s]]
        epsilonTemp = sum(error)

        # 判断如果错误率大于0.5，则终止本轮训练，进入下一轮
        while (epsilonTemp > 0.5):
            # 初始化样本权值为为泊松分布，并进行标准化到1
            weight = [-math.log(random.random() * 999 / 1000) for j in range(sampleNumber)]
            sumWeight = sum(weight)
            weightStd = [weight[p] / sumWeight for p in range(sampleNumber)]
            # 根据样本权值，记录每个样本在数轴[0,1]范围内所对应的区间，以方便根据随机数来产生对应样本
            weightStdsum = [sum(weightStd[:q]) for q in range(1, sampleNumber)]
            weightStdsum.append(1)
            # 轮次加1
            k = k + 1
            ### 重新根据样本权重生成样本集合 ###
            newInputs_X = []
            newInputs_y = []
            for x in range(sampleNumber):
                # 根据样本权值，记录每个样本在数轴[0,1]范围内所对应的区间，以方便根据随机数来产生对应样本
                value = random.random()
                index = FindIndex(value, weightStdsum, 0, sampleNumber - 1)
                newInputs_X.append(X_train.iloc[index][:])
                newInputs_y.append(y_train.iloc[index])
            # 分类器训练
            Dtc = DecisionTreeClassifier()
            Dtc.fit(newInputs_X, newInputs_y)
            # 计算当前分类器的分类错误率
            y_predict = Dtc.predict(newInputs_X)
            error = [weightStd[s] for s in range(sampleNumber) if y_predict[s] != newInputs_y[s]]
            epsilonTemp = sum(error)

        # 判断如果错误率等于0，则终止本轮训练，进入下一轮
        if (epsilonTemp == 0):
            # 讲beta设为一个很小但部位0 的值，防止后续计算报错
            betaTemp = math.pow(10, -10)
            # 初始化样本权值为为泊松分布，并进行标准化到1
            weight = [-math.log(random.random() * 999 / 1000) for j in range(sampleNumber)]
            sumWeight = sum(weight)
            # 根据样本权值，记录每个样本在数轴[0,1]范围内所对应的区间，以方便根据随机数来产生对应样本
            weightStd = [weight[p] / sumWeight for p in range(sampleNumber)]
            weightStdsum = [sum(weightStd[:q]) for q in range(1, sampleNumber)]
            weightStdsum.append(1)
            # 轮次加1
            k = k + 1
        else:
            # 计算当前学习器评估指标beta
            betaTemp = epsilonTemp / (1 - epsilonTemp)
            # 根据当前预测结果调整样本权值
            for u in range(sampleNumber):
                if (y_predict[u] != newInputs_y[u]):
                    weightStd[u] = weightStd[u] / (2 * epsilonTemp)
                else:
                    weightStd[u] = weightStd[u] / (2 * (1 - epsilonTemp))
                if (weightStd[u] < math.pow(10, -8)):
                    weightStd[u] = math.pow(10, -8)
        # 记录每一轮训练的错误率，beta，以及分类器
        epsilon.append(epsilonTemp)
        beta.append(betaTemp)
        C.append(Dtc)
        print("Epoch", i, "done!")

        # 对迭代次数 T = 5,10,15...50 时，分别记录下对应的训练时间和分类精度
        Ite = np.linspace(5, 50, 10, dtype=int)
        if i in Ite:
            # 记录预测样本所耗时间
            predict_timeStart = time.time()
            # 记录所有输出类别集合
            outputs = ['draw', 'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven',
                       'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen']
            # 最终所分的类别
            y_predictFinal = []
            # 根据所有训练的分类器，对每个样本进行分类预测
            for j in range(len(X_test)):
                # 记录样本分到每个类别对应的得分，也即是被分为每一个类别的置信度
                sumScore = np.zeros(18)
                for h in range(i):
                    # 每个分类器的分类结果
                    y_predictTemp = C[h].predict([X_test.iloc[j][:]])
                    for m in range(18):
                        if (outputs[m] == y_predictTemp):
                            # 更新计算样本对应类别的置信度
                            sumScore[m] = sumScore[m] + math.log((1 / beta[h]))
                            break
                # 最大置信度所对应的类别即为所分类别
                sumScore = sumScore.tolist()
                max_index = sumScore.index(max(sumScore))
                y_predictFinal.append(outputs[max_index])
            # 计算分类精度
            accuracy = metrics.accuracy_score(y_test, y_predictFinal)
            Accuracys.append(accuracy)
            # 记录训练时间，需要减去预测样本所耗费的时间
            predict_timeEnd = time.time()
            predict_time+=predict_timeEnd - predict_timeStart
            endTime = time.time()
            CostTimes.append(endTime - startTime-predict_time)
    # 打印结果
    print("Accuracy:", Accuracys)
    print("Cost time:", CostTimes)
    # 绘制迭代次数——分类精度关系图
    plt.figure('分类精度')
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.title("MultiBoosting迭代次数——分类精度关系图")
    plt.xlabel("迭代次数")
    plt.ylabel("分类精度")
    plt.plot(Ite, Accuracys, label='MultiBoosting')
    plt.legend(loc='lower right')

    # 绘制迭代次数——运行时间关系图
    plt.figure('训练时间')
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.title("MultiBoosting迭代次数——运行时间关系图")
    plt.plot(Ite, CostTimes, label='MultiBoosting')
    plt.legend(loc='lower right')
    plt.xlabel("迭代次数")
    plt.ylabel("运行时间")
    plt.show()

