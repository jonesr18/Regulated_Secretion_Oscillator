from scipy.integrate.odepack import odeint
from pylab import fft, fftfreq
from numpy import linspace, meshgrid, array, zeros, size, isnan
from random import uniform

'''
Organizes and runs the dual cell secretion oscillator model.

Created on Nov 26, 2014

@author: Ross Jones
@contact: jonesr18@outlook.com
@organization: MIT Course 20.405
@version: 1.1
'''
class ModelODE(object):
    '''
    Manages the ODE model for this project
    '''
    
    # Fields
    params = []
    init = []
    names = []
    y0 = []
    p = []

    def __init__(self, init = None, params = None):
        '''
        Constructor
        '''
        
        # Get params, initial conditions, and species names from setup
        self.params, self.init, self.names = self._setup()
        
        if not params == None:
            self.params = params
        if not init == None: 
            self.init = init
    
    
    def diffEQs(self, init, t, params, ignore = 'ignore'):
        '''
        ODE model for the system
        '''
        
        # Unpack params
        ktxn, ktln, kdm, kexport, KCa, N, M, P, kdp, kdilute, \
            kinflux, koutflux, KAct, KRep = params
        
        # Unpack species
        init[isnan(init)] = 0
        ActDNA, ActRNA, ActProt, ActEC, ActChannel, ActCaC, \
            RepDNA, RepRNA, RepProt, RepEC, RepChannel, RepCaC, CaEC = init
        
        # Ligand production
        dActDNA = 0
        dRepDNA = 0
        dActRNA = ktxn * ActDNA - (ktln + kdm) * ActRNA
        dActProt = ktln * ActRNA - (kexport * ActCaC**N / (KCa**N + ActCaC**N) + kdp) * ActProt
        dActEC = kexport * ActCaC**N / (KCa**N + ActCaC**N) * ActProt - kdilute * ActEC
        dRepRNA = ktxn * RepDNA - (ktln + kdm) * RepRNA
        dRepProt = ktln * RepRNA - (kexport * RepCaC**N / (KCa**N + RepCaC**N) + kdp) * RepProt
        dRepEC = kexport * RepCaC**N / (KCa**N + RepCaC**N) * RepProt - kdilute * RepEC
        
        # Calcium Flux
        dCaEC = 0
        dActChannel = 0
        dRepChannel = 0
        channelActivation = ActEC**M / (KAct**M + ActEC**M) * KRep**P / (KRep**P + RepEC**P)
        dActCaC = kinflux * channelActivation * ActChannel * CaEC - koutflux * (ActCaC - 100e-3)
        dRepCaC = kinflux * channelActivation * RepChannel * CaEC - koutflux * (RepCaC - 100e-3)
        
        # Collect species changes
        dSpecies = array([
            dActDNA, dActRNA, dActProt, dActEC, dActChannel, dActCaC,
            dRepDNA, dRepRNA, dRepProt, dRepEC, dRepChannel, dRepCaC, dCaEC ])
        
        # Ensure no species go below 0
        # Wherever the resulting [] will be negative, reduce the change by the overshoot. 
        dSpecies[isnan(dSpecies)] = 0 
        dInit = array(init) + dSpecies
        dInit[dInit >= 0] = 0
        return dSpecies - dInit
        
    
    def runModel(self, tspan, vary = False):
        '''
        Runs the ODE model for the system
        '''
        
        if vary:
            # self.y0 = self.varyVector(self.init)
            self.y0 = self.init
            self.p = self.varyVector(self.params)
        else:
            self.y0 = self.init
            self.p = self.params
        
        # print "tspan: ", tspan
        # print "y0: ", y0
        # print "p: ", p
        
        return odeint(self.diffEQs, self.y0, tspan, (self.p, 'annoyingly necessary value'))
    
    
    def varyVector(self, vector):
        '''
        Varies and returns the items in a given vector.
        
        newVector === Python array
        
        Vector items are varied uniformly in log space +/- 10^2 from their original value
        '''
        
        newVector = []
        for item in vector:
            varItem = item * 10**uniform(-2, 2)
            if isnan(varItem):
                varItem = 0
            newVector.append(varItem)
        
        return newVector
    
    
    def calcFourier(self, yout, tspan):
        '''
        Calculate FFT and return magnitude spectrum and frequencies
        
        freqs === Python array
        aMags === numpy array
        rMags === numpy array
        '''
        ActEC = yout[:, 3]
        RepEC = yout[:, 9]
        freqs = fftfreq(len(tspan), tspan[1] - tspan[0])
        aMags = array(abs((fft(ActEC))))
        rMags = array(abs((fft(RepEC))))
        return freqs, aMags, rMags
    
    
    def plotQuiver(self, yout, y0, axes):
        '''
        Plots a phase diagram of the activator and repressor ligands
        '''
        ActEC = yout[:, 3]
        RepEC = yout[:, 9]
        ActSpace = linspace(min(ActEC), max(ActEC), 20)
        RepSpace = linspace(min(RepEC), max(RepEC), 20)
        X, Y = meshgrid(ActSpace, RepSpace)
        cols = size(X, 0)
        rows = size(X, 1)
        u = zeros([cols, rows])
        v = zeros([cols, rows])
        for i in range(0, cols):
            for j in range(0, rows):
                y0[3] = X[i, j]
                y0[9] = Y[i, j]
                dy = self.diffEQs(y0, 0, self.params)
                u[i, j] = dy[3]
                v[i, j] = dy[9]
        axes.quiver(X, Y, u, v)
    
    
    def _setup(self):
        ''' 
        Sets up the parameters, initial conditions, and species names, then returns them
        '''
        
        # (units: s, uM, copies)
    
        # Parameters 
        ktxn = 0.05         # Ligand transcription rate                     | uM s^-1
        ktln = 1.5          # Ligand translation rate                       | s^-1
        kdm = 0.005         # Ligand mRNA degradation rate                  | s^-1
        kexport = 35e+3     # Rate of ligand export from cell               | s^-1
        KCa = 15            # Kd for Ca+2-mediated membrane fusion          | uM
        N = 5               # Cooperativity of Ca+2 w/ fusion complex       | 
        M = 1               # Cooperativity of activator with channel       | 
        P = 1               # Cooperativity of blocker with channel         | 
        kdp = 1e-4          # Ligand protein degradation rate               | s^-1
        kdilute = 1000      # Extracellular dilution rate of ligand         | s^-1
        kinflux = 50        # Rate of Ca+2 influx from open channels        | s^-1
        koutflux = 50       # Rate of Ca+2 outflux from cell                | s^-1
        KAct = 45e-3        # Kd for activator binding to channel           | uM
        KRep = 45e-3        # Kd for blocker binding to channel             | uM
        
        # Initial Conditions
        ActDNA = 5          # Activator ligand DNA                          | Copies
        ActRNA = 0          # Activator ligand mRNA                         | uM
        ActProt = 0         # Activator ligand protein (in granules)        | uM
        ActEC = 60e-3       # Activator extracellular ligand                | uM
        ActChannel = 35     # Activator cell Ca+2 channels (for influx)     | Copies
        ActCaC = 100e-3     # Activator cell cytoplasmic Ca+2               | uM
        RepDNA = 5          # Repressor ligand DNA                          | Copies
        RepRNA = 0          # Repressor ligand mRNA                         | uM
        RepProt = 0         # Repressor ligand protein (in granules)        | uM
        RepEC = 0           # Repressor extracellular ligand                | uM
        RepChannel = 35     # Repressor cell Ca+2 channels (for influx)     | Copies
        RepCaC = 100e-3     # Repressor cell cytoplasmic Ca+2               | uM
        CaEC = 1.8e+3       # Extracellular Ca+2                            | uM
        
        # Collect parameters and initial conditions
        params = [
            ktxn, ktln, kdm, kexport, KCa, N, M, P, kdp, kdilute, 
            kinflux, koutflux, KAct, KRep] 
        init = [
            ActDNA, ActRNA, ActProt, ActEC, ActChannel, ActCaC,
            RepDNA, RepRNA, RepProt, RepEC, RepChannel, RepCaC, CaEC]
        names = ['Activator_{DNA}', 'Activator_{mRNA}', 'Activator_{Prot}', 'Activator_{EC}',
                 'Activator Cell Channel', 'Activator Ca_C',
                 'Repressor_{DNA}', 'Repressor_{mRNA}', 'Repressor_{Prot}', 'Repressor_{EC}',
                 'Repressor Cell Channel', 'Repressor Ca_C', 'Ca_{EC}']
        
        return params, init, names
    
    
    
    
