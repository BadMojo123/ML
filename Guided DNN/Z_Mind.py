import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt

from myconfig import *


def calculateProfit1(test_x, Result):
    i, j = Result.shape
    in_feature = 1
    AccountBalance = 0
    wins = 0
    trans = 1
    Open = 0
    direction = ""

    # save for testing
    # FileName = "data/Test_x.csv"
    # np.savetxt(FileName, test_x, delimiter=",")
    # FileName = "data/Result.csv"
    # np.savetxt(FileName, Result, delimiter=",")

    for li in range(0, i - in_feature - 1):
        if Result[li, 0] == 1:  # BUY
            if (direction == "SELL"):
                AccountBalance = AccountBalance + (Open - test_x[li, -4])
            if (direction == "BUY"):
                continue
            trans += 1
            Open = test_x[li, -4]
            direction = "BUY"
            continue
        if Result[li, 2] == 1:  # SELL
            if (direction == "BUY"):
                AccountBalance = AccountBalance - (Open - test_x[li, -4])
            if (direction == "SELL"):
                continue
            trans += 1
            Open = test_x[li, -4]
            direction = "SELL"
            continue
        if Result[li, 1] == 1:  # CLOSE SELL
            if (direction == "BUY"):
                continue
            if (direction == "SELL"):
                AccountBalance = AccountBalance + (Open - test_x[li, -4])
                direction = "CLOSE"
        if Result[li, 3] == 1:  # CLOSE BUY
            if (direction == "SELL"):
                continue
            if (direction == "BUY"):
                AccountBalance = AccountBalance - (Open - test_x[li, -4])
                direction = "CLOSE"

    return AccountBalance, trans


def calculateProfit2(test_x, Result):
    i, j = Result.shape
    in_feature = 1
    AccountBalance = 0
    trans = 1
    Open = 0
    direction = ""
    for li in range(0, i - in_feature - 1):
        if Result[li, 0] == 1:  # BUY
            if (direction == "SELL"):
                AccountBalance = AccountBalance + (Open - test_x[li, -4])
                # AccountBalance= AccountBalance + (test_x[li,-2] - test_x[li,-4])
            if (direction == "BUY"):
                continue
            trans += 1
            Open = test_x[li, -4]
            direction = "BUY"
            continue
        elif Result[li, 2] == 1:  # SELL
            if (direction == "BUY"):
                AccountBalance = AccountBalance - (Open - test_x[li, -4])
            if (direction == "SELL"):
                continue
            trans += 1
            Open = test_x[li, -4]
            direction = "SELL"
            continue
        else:
            if (direction == "SELL"):
                AccountBalance = AccountBalance + (Open - test_x[li, -4])
                direction = "CLOSE"
            if (direction == "BUY"):
                AccountBalance = AccountBalance - (Open - test_x[li, -4])
                direction = "CLOSE"

    return AccountBalance, trans


def plotResult(Data, prediction, testSet):
    i, j = prediction.shape
    predictionPlot = np.zeros((i, j))
    t = np.arange(0, i, 1)

    s = Data[testSet:, -4]
    fig, ax = plt.subplots()

    for li in range(0, i):
        for lj in range(0, j):
            temp = max(prediction[li, :])
            if prediction[li, lj] == temp:
                predictionPlot[li, :] = [None, None, None, None]
                predictionPlot[li, lj] = s[li]
                continue

    ax.plot(t, s, 'b-', t, predictionPlot[:, 0], 'g*')
    ax.plot(t, predictionPlot[:, 1], 'g.')
    ax.plot(t, predictionPlot[:, 2], 'r.')
    ax.plot(t, predictionPlot[:, 3], 'r*')

    ax.grid()
    fig.savefig("plots/hist.png")
    plt.show()
    return


# def sigmoid(x):
#     X_std=(x-EURUSDMIN)/(EURUSDMAX-EURUSDMIN)
#     return X_std
#     return x

def LoadFile(FILE_NAME):
    # Day(0)        weekday(1)      Hour(2)  minute(3)   size(4)   H(5)    L(6)    C(7)    Volume(8)
    RawData = genfromtxt(FILE_NAME, delimiter=',')
    return RawData


def AddLabels(Data, HistReq=62):
    i, j = Data.shape
    # pick = (0.00045)/(EURUSDMAX - EURUSDMIN)
    pick = 0.00015
    fee = 0.00001
    Labels = np.zeros((i, 4))
    # Label dictionary:
    # [1,0,0,0] - buy
    # [0,1,0,0] - up/close sell
    # [0,0,1,0] - Sell
    # [0,0,0,1] - down/close buy
    for li in range(0, i - 2):
        if (Data[li + 1, 5] > Data[li, 5] + pick or Data[li + 2, 5] > Data[li, 5] + pick):
            Labels[li, 0] = 1
        if (Data[li + 1, 7] > Data[li, 7] + fee):
            Labels[li, 1] = 1
        if (Data[li + 1, 6] < Data[li, 6] - pick or Data[li + 2, 6] < Data[li, 6] - pick):
            Labels[li, 2] = 1
        if (Data[li + 1, 7] < Data[li, 7] - fee):
            Labels[li, 3] = 1

    NormData = np.zeros((i, 18))

    for li in range(HistReq, i):
        NormData[li, :] = Norm(Data, li)

    return Data[HistReq:-2, :], Labels[HistReq:-2, :], NormData[HistReq:-2, :]


def Norm(data, li=0):
    # Day(0)        weekday(1)      Hour(2)  minute(3)   size(4)   H(5)    L(6)    C(7)    Volume(8)

    if (li == 0):
        li, j = data.shape
        li = li - 1

    NormalizedRow = np.zeros((0, 18))

    NormalizedRow = [
        data[li, 0] / 31,
        data[li, 1] / 7,
        data[li, 2] / 24,
        data[li, 3] / 60,
        data[li, 4] / EURUSDMAXSIZE_1M,  # Normalized Size
        (data[li, 5] - data[li, 7]) / EURUSDMAXSIZE_1M,  # high
        (data[li, 6] - data[li, 7]) / EURUSDMAXSIZE_1M,  # Low
        (data[li, 7] - data[li - 1, 7]) / EURUSDMAXSIZE_1M,  # Change in last 1M
        (data[li, 8]) / (EURUSDVOLMAX),  # Normalized Volume
        sum(data[li - 5:li, 8]) / (5 * EURUSDVOLMAX),  # mean volume 5 min
        sum(data[li - 10:li, 8]) / (10 * EURUSDVOLMAX),  # mean volume 10 min
        (data[li, 7] - data[li - 10, 7]) / (EURUSDMAXSIZE_1M),  # mean change 10 min
        (data[li, 7] - data[li - 30, 7]) / (EURUSDMAXSIZE_1M),  # mean Change 30 min
        (data[li, 7] - data[li - 60, 7]) / (EURUSDMAXSIZE_1M),  # mean change 60 min
        np.std(data[li - 10:li, 7]),
        np.std(data[li - 30:li, 7]),
        np.std(data[li - 60:li, 7]),
        (data[li, 7] - min(data[li - 14:li, 6])) / (max(data[li - 14:li, 5]) - min(data[li - 14:li, 6]))
        # Stochactic oscilator
    ]
    return NormalizedRow


def SelectData(testSet, x_full, Labels):
    Train_x = x_full[0:testSet, :]
    Train_y = Labels[0:testSet, ]
    Test_x = x_full[testSet:, :]
    Test_y = Labels[testSet:, ]
    return Train_x, Train_y, Test_x, Test_y


def stock_sentiment(quote, num_tweets):
    # Checks if the sentiment for our quote is
    # positive or negative, returns True if
    # majority of valid tweets have positive sentiment
    # First we login into twitter
    import tweepy
    from textblob import TextBlob

    consumer_key = '6cYrWdkqMoTmHrETx1N5GBDzo'
    consumer_secret = 'CaqAbL4NP02VMRLHjYv2tEwTQkFL84HMYk2EagLJfZAGDwQi7c'
    access_token = '986243669291388929-u4aG0BhCloJIPVrcbUMAZzOO6PSZv3g'
    access_token_secret = '7H5mRCuTFNT4kx7KP77LndsK2wiLfM6s4VwGiEvvha79p'
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    user = tweepy.API(auth)

    list_of_tweets = user.search(quote, count=num_tweets)
    positive, null = 0, 0

    for tweet in list_of_tweets:
        blob = TextBlob(tweet.text).sentiment
        if blob.subjectivity == 0:
            null += 1
            next
        if blob.polarity > 0:
            positive += 1

    if positive > ((num_tweets - null) / 2):
        return True

    return False
