# https://github.com/VincentRagusa/binTools

#this function returns the optimal number of histogram bins to display the input data
def numBins(data):
    from statistics import stdev
    from numpy import sqrt, pi
    STD = stdev(data)
    if STD == 0:
        return 1
    LEN = len(data)
    RNG = max(data)-min(data)
    B =  (24*sqrt(pi))**(1/3) * STD * LEN**(-1/3)
    M = round(RNG/B)
    return M

def kernel(u):
    return (3/4)*(1-u**2) if -1 <= u <= 1 else 0

def F(data,h,t):
    return (1/(len(data)*h))*sum([kernel((t-x)/h) for x in data])

#this function returns a curve that fits your data 
#it is not a regression, it simply maps the outline of a histogram
def kernelFit(data):
    from numpy import arange
    from statistics import stdev
    low = min(data)
    high = max(data)
    s = stdev(data)
    h = 1.06*s*(len(data)**(-1/5))
    r = high-low
    curve = [[t,F(data,h,t)] for t in arange(low-0.05*r,high+0.05*r,r/100)]
    return list(zip(*curve))

if __name__ == "__main__":
    #here is a tutorial of sorts, it plots both the histogram and the fit-curve
    import numpy as np
    import matplotlib.pyplot as plt
    
    #make some fake data
    testData = np.random.default_rng().normal(0, 1, 1000)
    
    #test the curve fitting
    fitX, fitY = kernelFit(testData)
    plt.plot(fitX,fitY)
    
    #test the histogram with optimal bins
    plt.hist(testData,bins=numBins(testData),density=True)
    
    plt.show()