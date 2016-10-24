import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation


def load_data(file_name, sequence_length=10, split=0.8):
    df = pd.read_csv(file_name, sep=',', usecols=[1])  # 读取csv文件的 第二列
    data_all = np.array(df).astype(float)
    scaler = MinMaxScaler()
    data_all = scaler.fit_transform(data_all)  # 把数据缩放到0-1之间

    data = []
    for i in range(len(data_all) - sequence_length - 1):
        x = data_all[i: i + sequence_length + 1]  # sequence_length=10 ， 但要把数据集分成每个11个记录
        data.append(x)

    reshaped_data = np.array(data).astype('float64')  # 看不出啥效果
    np.random.shuffle(reshaped_data)  # 把这个数据随机打乱

    # 对x进行统一归一化，而y则不归一化
    x = reshaped_data[:, :-1]
    y = reshaped_data[:, -1]

    # 分割训练和测试
    split_boundary = int(reshaped_data.shape[0] * split)
    train_x = x[: split_boundary]
    test_x = x[split_boundary:]

    train_y = y[: split_boundary]
    test_y = y[split_boundary:]

    return train_x, train_y, test_x, test_y, scaler


def build_model():
    # input_dim是输入的train_x的最后一个维度，train_x的维度为(n_samples, time_steps, input_dim)
    model = Sequential()
    model.add(LSTM(input_dim=1, output_dim=50, return_sequences=True))
    model.add(LSTM(output_dim=10, return_sequences=False))
    model.add(Dense(output_dim=1, activation='linear'))
    model.compile(loss='mse', optimizer='rmsprop')
    return model


def train_model(train_x, train_y, test_x, test_y):
    model = build_model()
    model.fit(train_x, train_y, batch_size=512, nb_epoch=30, validation_split=0.1, verbose=2)  # 训练模型
    predict = model.predict(test_x)  # 预测 测试集
    print('predict -- ' * 5)
    print(predict)
    predict = np.reshape(a=predict, newshape=(predict.size, ))  # 数组格式转换 [[v],[v],[v]] --》 [v,v,v]
    print('reshape predict -- ' * 5)
    print(predict)
    print('test_y -- ' * 5)
    print(test_y)

    try:
        fig = plt.figure(1)
        plt.plot(predict, 'r:')
        plt.plot(test_y, 'g-')
        plt.legend(['predict', 'true'])
    except Exception as e:
        print(e)
    return predict, test_y


if __name__ == '__main__':

    train_x, train_y, test_x, test_y, scaler = load_data('international-airline-passengers.csv')

    print('-' * 20)
    print(test_x)

    print(train_x.shape)
    # (106, 10, 1)
    # 106个训练集
    # 每个训练集 有10行数据
    # 每行数据 有1个维度

    exit()
    train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))
    test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], 1))

    print('-'*20)
    print(test_x)


    predict_y, test_y = train_model(train_x, train_y, test_x, test_y)
    predict_y = scaler.inverse_transform([[i] for i in predict_y])  # 反变换

    test_y = scaler.inverse_transform(test_y)
    fig2 = plt.figure(2)
    plt.plot(predict_y, 'g:')
    plt.plot(test_y, 'r-')
    plt.show()

