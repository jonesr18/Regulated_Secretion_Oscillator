from pylab import subplots, xlabel, ylabel, legend, show
from numpy import array, shape, linspace, isnan, logical_and, seterr, size, mean, \
    median, corrcoef, zeros, std, min, max
import model as mv1
import modelV2 as mv2
from csv import reader, writer
from time import clock
import os
from numpy.fft.helper import fftfreq
from scipy.sparse.linalg import eigs
from math import floor, ceil, sqrt
seterr(divide = 'ignore', invalid = 'ignore')

'''
Script to run the model and analyze data.

Created on Nov 26, 2014

@author: Ross Jones
@contact: jonesr18@outlook.com
@organization: MIT Course 20.405
''' 

def testRun(m, tspan):
    '''
    Method mostly to test plotting and running abilities
    '''
    yout = m.runModel(tspan)
    
    # Plot species concentrations over time
    fig, axes = subplots()
    axes.plot(tspan, yout)
    xlabel('Time (sec)')
    ylabel('Concentration (uM) or Copies')
    legend(m.names)
    
    # Plot phase diagram of ActEC and RepEC
    fig, axes2 = subplots()
    m.plotQuiver(yout, axes2)


def iterTestRun(m, tspan):
    '''
    Method to perform iterations and generate plots
    '''
        
    # Test 16 random conditions - plot fft and phase diagrams
    fig, axes = subplots(4, 4)
    fig, axes2 = subplots(4, 4)
    for i in range(0, 4):
        for j in range(0, 4):
            # Run model
            yout = m.runModel(tspan, True)
            
            # Plot quiver diagram
            m.plotQuiver(yout, axes[i, j])
            
            # Plot Fourier transform
            freqs, aMags, rMags = m.calcFourier(yout, tspan)
            
            # Plot FFT
            axes2[i, j].plot(freqs, aMags, freqs, rMags)
            if i == 0:
                xlabel('Frequency (s^-1)')
            if j == 3:
                ylabel('Magnitude')
            
    legend(['Activator', 'Repressor'])


def monteCarloRun(m, tspan):
    '''
    Run Monte Carlo simulation. Select top 16 0-10 Hz freqs
    '''
    
    # Run simulations
    niter = 5000
    youts = []
    aMags = []
    rMags = []
    y0 = []
    p = []
    sums = []
    start1 = clock()
    for i in range(0, niter):
        yout = m.runModel(tspan, True)
        f, a, r = m.calcFourier(yout, tspan)
        
        # Find frequencies between 0 and 10 Hz
        if i == 0:
            freqs = f
            f = array(f)
            idx = logical_and(f > 0.2, f < 3)    # Logical index
        
        # Determine if result passes criteria
        thresh = 100
        magSum = sum(a[idx]) + sum(r[idx])
        if not isnan(magSum) and sum(a[idx]) > thresh and sum(r[idx]) > thresh:
            youts.append(yout)
            aMags.append(a)
            rMags.append(r)
            y0.append(m.y0)
            p.append(m.p)
            sums.append(magSum)
        
        print "Finished Iteration", i + 1
    
    end1 = clock()
    print "Simulation loop time:", end1 - start1
    
    # Extract highest sum
    sums = array(sums)
    extract = array(sorted(enumerate(sums), key = lambda s: s[1], reverse = True))
    
    # Create plots
    youts = array(youts)
    aMags = array(aMags)
    rMags = array(rMags)
    y0 = array(y0)
    p = array(p)
    fig, axes1 = subplots(4, 4)
    fig, axes2 = subplots(4, 4)
    fig, axes3 = subplots(4, 4)
    k = 0
    start2 = clock()
    for i in range(0, 4):
        for j in range(0, 4):
            
            # Check if index too big
            if k >= size(extract, 0):
                break
            
            # Recover index from extract vector
            idx = extract[k, 0]
            
            # Plot quiver
            m.plotQuiver(youts[idx], y0[idx], axes1[i, j])
            
            # Plot FFT
            axes2[i, j].plot(freqs, aMags[idx], freqs, rMags[idx])
            axes2[i, j].set_yscale('log')
            
            # Plot Species []
            specPlot = [10, 11]
            axes3[i, j].plot(tspan, youts[idx, :, specPlot].T)
            axes3[i, j].set_yscale('log')
            legend(array(m.names)[specPlot], loc = 0)
            
            # Update index
            k += 1
            
    end2 = clock()
    print "Plot loop time:", end2 - start2
    
    return youts, aMags, rMags, y0, p, extract


def analyzeData(youts, aMags, rMags, y0, p, m):
    '''
    Analyzes data which was saved to CSV output.
    '''
    
    # Find frequencies between 0.2 and 3 Hz
    freqs = fftfreq(len(tspan), tspan[1] - tspan[0])
    f = array(freqs)
    idx = logical_and(f > 0.2, f < 3)    # Logical index
    
    # Find best responses where the frequencies are high between range above
    sums = [sum(aMags[i, idx]) + sum(rMags[i, idx]) for i in range(0, size(aMags, 0))]
    sums = array(sums)
    extract = array(sorted(enumerate(sums), key = lambda s: s[1], reverse = True))
    
    best100 = [int(i) for i in extract[0:100, 0]]
    stats = calcStats(p[best100])
    for i in range(0, len(m.pnames)):
        print m.pnames[i] + ':', stats['lower95'][i], '|', stats['upper95'][i], '|', stats['median'][i]
    
    '''
    
    # Compute partial correlation coefficient
    R = corrcoef(p[best100].T, sums[best100].T)
    R[isnan(R)] = 0
    R_F = R[-1, 0:-1]
    print ['{:0.2f}'.format(i) for i in R_F]
    fig, axes = subplots()
    axes.bar(range(0, len(R_F)), R_F)
    axes.set_xticks(linspace(0.4, len(R_F) - 0.6, len(R_F)))
    axes.set_xticklabels(m.pnames, rotation = 60)
    
    \'''
    
    # Run longer simulations
    tspanNew = linspace(0, 100, 10000)
    fig, axes = subplots(4, 4)
    k = 0
    for i in range(0, 4):
        for j in range(0, 4):
            
            # Recover index from extract vector
            run = extract[k, 0]
            
            # Run simulation
            m = mv2.ModelODE()
            youtNew = m.runModel(tspanNew, True)
            
            # Plot results
            specPlot = [10, 11]
            axes[i, j].plot(tspanNew, youtNew[:, specPlot])
            axes[i, j].set_yscale('log')
            legend(array(m.names)[specPlot], loc = 0)
            
            # Update index
            k += 1
            print 'Finished iteration', k 
    '''
    

def writeCSV(filename, data):
    '''
    Writes the given data to a CSV file in the current directory with the given name.
    '''
    
    # Check input
    if not filename[-4:-1] == '.cs':
        filename += '.csv'
    
    cwd = os.getcwd()
    csvFile = open(cwd + '\\' + filename, 'w')
    w = writer(csvFile)
    w.writerows(data)
    csvFile.close()


def openCSV(filename):
    '''
    Reads and returns a data array from the CSV file in the current directory with the given name.
    '''
    
    # Check input
    if not filename[-4:-1] == '.cs':
        filename += '.csv'
    
    cwd = os.getcwd()
    csvFile = open(cwd + '\\' + filename, 'r')
    w = reader(csvFile)
    data = []
    for row in w:
        if row[0][0] == '[':
            # 3-D data was stored
            rowData = []
            for col in row:
                colData = col[1:-1].split()     # Get rid of brackets and create array of values
                colData = [float(i) for i in colData]   # Convert strings to doubles
                rowData.append(colData)
            data.append(rowData)
        else:
            # 1/2-D data was stored
            row = [float(i) for i in row]   # Convert strings to doubles
            data.append(row)
    csvFile.close()
    return data
    
    
def calcStats(data):
    '''
    Computes the 95% confidence interval of the given data. The data should be given 
    as a numpy array where each row is an observation and each col is a parameter.
    
    The return is a dictionary with the following keys:
        mean
        median
        max
        min
        lower95        (lower 95% confidence interval value)
        upper95        (upper 95% confidence interval value)
        stdev          (standard deviation)
        SEM            (standard error)
    '''
    nObs = size(data, 0)
    rangeCI = floor(0.95 * nObs)
    
    if not rangeCI % 2 == nObs % 2:
        rangeCI = ceil(0.95 * nObs)
    
    spacer = int((nObs - rangeCI) / 2)
    sortedData = array([sorted(data[:, i]) for i in range(size(data, 1))]).T
    return {
        'mean': mean(data, 0),
        'median': median(data, 0),
        'max': max(data, 0),
        'min': min(data, 0),
        'lower95': sortedData[spacer, :],
        'upper95': sortedData[-spacer - 1, :],
        'stdev': std(data, 0),
        'SEM': std(data, 0) / sqrt(nObs) }
    
    
def makePoly(coeffs, tspan):
    '''
    Returns the response of a polynomial with given coeffs at the given time points.
    '''
    output = []
    for t in tspan:
        total = 0
        for i, coeff in enumerate(coeffs):
            total += coeff * t**(len(coeffs) - i - 1)
        output.append(total)
    return output
    
    
if __name__ == '__main__':
    
#     Setup model
#     m = mv1.ModelODE()
    tspan = linspace(0, 10, 1000)
    
#     Run model
#     testRun(m, tspan)
#     iterTestRun(m, tspan)
#     youts, aMags, rMags, y0, p, extract = monteCarloRun(m, tspan)
    
    '''
    y0 = None
    extract = None
    for i in range(0, 1):
        if i == 0:
            m = mv2.ModelODE()
        else:
            m = mv2.ModelODE(y0[extract[0, 0]])
        youts, aMags, rMags, y0, p, extract = monteCarloRun(m, tspan)
    
    start = clock()
    writeCSV('youts.csv', youts)
    writeCSV('aMags.csv', aMags)
    writeCSV('rMags.csv', rMags)
    writeCSV('y0.csv', y0)
    writeCSV('p.csv', p)
    end = clock()
    print "TIme to save:", end - start
    
    '''
#     youts = array(openCSV('youts.csv'))
    aMags = array(openCSV('aMags'))
    rMags = array(openCSV('rMags'))
    y0 = array(openCSV('y0'))
    p = array(openCSV('p'))
    
    m = mv2.ModelODE()
    
#     analyzeData(None, aMags, rMags, y0, p, m)
    
    A = m.jacobian()
    eigenVals, eigenVecs = eigs(A)
    
    print eigenVals
    
    '''
    
    tspanNew = linspace(0, 1000, 10000)
    
    for i in range(100):
        L, V = eigs(m.jacobian(True))
        Lr = L.real
        if any(Lr > 0):
            fig, axes = subplots()
            axes.plot(tspanNew, makePoly(Lr, tspanNew))
        Li = L.imag
        if any(Li > 0):
            fig, axes = subplots()
            axes.plot(Lr, Li)
    
    #'''
    show()
    
    
    
    