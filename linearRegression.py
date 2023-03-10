# Linear regression algorithm
def linearRegression(data):
    n = len(data)
    sumX = sum(data[x])
    sumY = sum(data[y])
    sumXtimesY = sum(data[x] * data[y])
    sumXtimesSumY = sum(data[x]) * sum(data[y])
    squaredSumX = sum(data[x]**2)
    sumXsquared = sum(data[x])**2

    num = (n * sumXtimesY) - sumXtimesSumY
    denom = (n * squaredSumX) - sumXsquared

    m = num / denom

    meanX = sum(data[x])/len(data[x])
    meanY = sum(data[y])/len(data[y])
    mTimesMeanX = m * meanX

    b = meanY - mTimesMeanX

    return {"m":m, "b":b}
