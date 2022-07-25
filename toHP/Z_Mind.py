import matplotlib.pyplot as plt
import numpy
from numpy import genfromtxt

EURUSDMIN = 1.00600
EURUSDMAX = 1.5
EURUSDVOLMIN = 0
EURUSDVOLMAX = 500
EURUSDMAXSIZE_1M = 0.0003


def calculateProfit(test_x, Result, testSet):
    i, j = Result.shape
    in_feature = 1
    AccountBalance = 0
    trans = 1
    for li in range(0, i - in_feature - 1):
        if Result[li, 0] == 1:
            trans += 1
            AccountBalance = AccountBalance + (test_x[li, j - 3] - test_x[li, -2])
            continue
        if Result[li, 2] == 1:
            trans += 1
            AccountBalance = AccountBalance - (test_x[li, -3] - test_x[li, -2])
            continue

    return AccountBalance, trans


def plotResult(Data, prediction, testSet):
    i, j = prediction.shape
    predictionPlot = numpy.zeros((i, j))
    t = numpy.arange(0, i, 1)

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


def sigmoid(x):
    X_std = (x - EURUSDMIN) / (EURUSDMAX - EURUSDMIN)
    return X_std
    return x


def LoadFile(FILE_NAME):
    # Day(0)        weekday(1)      Hour(2)  minute(3)   O(4)   H(5)    L(6)    C(7)    Volume(8)
    RawData = genfromtxt(FILE_NAME, delimiter=',')
    return RawData


def AddLabels(Data):
    i, j = Data.shape
    pick = (0.0001) / (EURUSDMAX - EURUSDMIN)
    fee = (0.00005) / (EURUSDMAX - EURUSDMIN)
    Labels = numpy.zeros((i, 4))

    # Label dictionary:
    # [1,0,0,0] - buy
    # [0,1,0,0] - close sell
    # [0,0,1,0] - Sell
    # [0,0,0,1] - close buy
    for li in range(0, i):
        if (Data[li, j - 1] > Data[li, j - 4] + pick):
            Labels[li] = [1, 0, 0, 0]
            continue
        elif (Data[li, j - 1] > Data[li, j - 4] + fee):
            Labels[li] = [0, 1, 0, 0]
            continue
        elif (Data[li, j - 1] < Data[li, j - 4] - pick):
            Labels[li] = [0, 0, 1, 0]
            continue
        elif (Data[li, j - 1] < Data[li, j - 4] - fee):
            Labels[li] = [0, 0, 0, 1]
            continue

    return Data[:, :-2], Labels


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
