from scipy.integrate.odepack import odeint
from pylab import fft, fftfreq
from numpy import linspace, meshgrid, array, zeros, size, isnan, seterr
from random import uniform
from time import clock
seterr(divide = 'ignore', invalid = 'ignore')

'''
Organizes and runs the dual cell secretion oscillator model.

Created on Nov 26, 2014

@author: Ross Jones
@contact: jonesr18@outlook.com
@organization: MIT Course 20.405
@version: 2.0
@change: Removed the mRNA species, added Ca-channel dynamics module
'''
class ModelODE(object):
    '''
    Manages the ODE model for this project
    '''
    
    # Fields
    params = []
    init = []
    names = []
    pnames = []
    y0 = []
    p = []

    def __init__(self, init = None, params = None):
        '''
        Constructor
        '''
        
        # Get params, initial conditions, and species names from setup
        self.params, self.init, self.names, self.pnames = self._setup()
        
        if not params == None:
            self.params = params
        if not init == None: 
            self.init = init
    
    
    def diffEQs(self, init, t, params, ignore = 'ignore'):
        '''
        ODE model for the system
        '''
        
#         start = clock()
        
        # Unpack params
        ktxn, ktln, kexport, KCa, N, kdp, kdilute, kinflux, koutflux, konA, \
            koffA, konR, koffR, konAR, koffAR = params
        
        # Unpack species
        init[isnan(init)] = 0
        init[init < 1e-9] = 0   # set any species < 1 fM to 0
        ActDNA, ActProt, ActCaC, \
            RepDNA, RepProt, RepCaC, \
            Channel, Channel_A, Channel_R, Channel_AR, \
            ActEC, RepEC, CaEC = init
        
#         end = clock()
#         print "Unpacking:", end - start
#         start = clock()
        
        # Ligand production
        dActDNA = 0
        dRepDNA = 0
#         try:
#             dActProt = ktxn * ktln * ActDNA - (kexport * ActCaC**N / (KCa**N + ActCaC**N) + kdp) * ActProt
#         except:
#             print "prot gen:", ktxn * ktln * ActDNA
#             # print "prot elim:", (kexport * ActCaC**N / (KCa**N + ActCaC**N) + kdp) * ActProt
#             print "KCa:", KCa
#             print "ActCaC", ActCaC
#             print "kexport", kexport
        dActProt = ktxn * ktln * ActDNA - (kexport * ActCaC**N / (KCa**N + ActCaC**N) + kdp) * ActProt
        dActEC = kexport * ActCaC**N / (KCa**N + ActCaC**N) * ActProt - kdilute * ActEC
        dRepProt = ktxn * ktln * RepDNA - (kexport * RepCaC**N / (KCa**N + RepCaC**N) + kdp) * RepProt
        dRepEC = kexport * RepCaC**N / (KCa**N + RepCaC**N) * RepProt - kdilute * RepEC
        
        # Channel dynamics
        dChannel = koffA * Channel_A + koffR * Channel_R - (konA * ActEC + konR * RepEC) * Channel
        dChannel_A = konA * Channel * ActEC - koffA * Channel_A - konAR * Channel_A * RepEC
        dChannel_R = konR * Channel * RepEC - koffR * Channel_R + koffAR * Channel_AR
        dChannel_AR = konAR * Channel_A * RepEC - koffAR * Channel_AR
        
        # Calcium Flux
        dCaEC = 0
        dActCaC = kinflux * Channel_A * CaEC - koutflux * (ActCaC - 100e-3)
        dRepCaC = kinflux * Channel_A * CaEC - koutflux * (RepCaC - 100e-3)
        
#         end = clock()
#         print "Calculations:", end - start
#         start = clock()
        
        # Collect species changes
        dSpecies = array([
            dActDNA, dActProt, dActCaC,
            dRepDNA, dRepProt, dRepCaC, 
            dChannel, dChannel_A, dChannel_R, dChannel_AR,
            dActEC, dRepEC, dCaEC ])
        
        # Ensure no species go below 0
        # Wherever the resulting [] will be negative, reduce the change by the overshoot.
        dSpecies[isnan(dSpecies)] = 0
        dInit = array(init) + dSpecies
        dInit[dInit > 0] = 0
        
#         end = clock()
#         print "Packing:", end - start 
        
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
                
        return odeint(self.diffEQs, self.y0, tspan, (self.p, 'annoyingly necessary value'))
    
    
    def varyVector(self, vector):
        '''
        Varies and returns the items in a given vector.
        
        newVector === Python array
        
        Vector items are varied uniformly in log space +/- 10^2 from their original value
        '''
        
        newVector = []
        for i, item in enumerate(vector):
            if i in [0, 1, 3, 4]:
                # Don't change ktxn, ktln, KCa, and N
                newVector.append(item)
            else:
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
        ActEC = yout[:, 10]
        RepEC = yout[:, 11]
        freqs = fftfreq(len(tspan), tspan[1] - tspan[0])
        aMags = array(abs((fft(ActEC))))
        rMags = array(abs((fft(RepEC))))
        return freqs, aMags, rMags
    
    
    def jacobian(self, vary = False):
        '''
        Creates and returns the jacobian matrix of the linearized system. 
        
        Includes the option of varying the parameters if desired.  
        '''
        
        # Check input
        if vary:
            P = self.varyVector(self.params)
        else:
            P = self.params
        
        # Unpack parameters
        ktxn, ktln, kexport, KCa, N, kdp, kdilute, kinflux, koutflux, konA, \
            koffA, konR, koffR, konAR, koffAR = P
        
        # Create and return Jacobian
        return array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [ktxn * ktln, 0, -(kexport + kdp), 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, ktxn * ktln, kexport, -kdilute, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, -(kexport + kdp), 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, kexport, -kdilute, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, -(koffA + koffR), koffA, koffR, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, koffA, -(koffA + koffAR), 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, koffR, 0, -koffR, koffAR, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, koffAR, 0, -koffAR, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0, 0, (kinflux + self.init[-1]), 0, -koutflux, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, (kinflux + self.init[-1]), 0, 0, -koutflux] ])
        
        
    def plotQuiver(self, yout, y0, axes):
        '''
        Plots a phase diagram of the activator and repressor ligands
        '''
        ActEC = yout[:, 10]
        RepEC = yout[:, 11]
        ActSpace = linspace(min(ActEC), max(ActEC), 20)
        RepSpace = linspace(min(RepEC), max(RepEC), 20)
        X, Y = meshgrid(ActSpace, RepSpace)
        cols = size(X, 0)
        rows = size(X, 1)
        u = zeros([cols, rows])
        v = zeros([cols, rows])
        for i in range(0, cols):
            for j in range(0, rows):
                y0[10] = X[i, j]
                y0[11] = Y[i, j]
                dy = self.diffEQs(y0, 0, self.params)
                u[i, j] = dy[10]
                v[i, j] = dy[11]
        axes.quiver(X, Y, u, v)
        axes.set_yscale('log')
    
    
    def _setup(self):
        ''' 
        Sets up the parameters, initial conditions, and species names, then returns them
        '''
        
        ### (units: s, uM, copies) ###
        
        # Parameters 
        ktxn = 0.05         # Ligand transcription rate                     | uM s^-1
        ktln = 1.5          # Ligand translation rate                       | s^-1
        kexport = 10e+3     # Rate of ligand export from cell               | s^-1
        KCa = 1.5           # Kd for Ca+2-mediated membrane fusion          | uM
        N = 5               # Cooperativity of Ca+2 w/ fusion complex       | 
        kdp = 1e-4          # Ligand protein degradation rate               | s^-1
        kdilute = 1000.0    # Extracellular dilution rate of ligand         | s^-1
        kinflux = 50.0      # Rate of Ca+2 influx from open channels        | s^-1
        koutflux = 50.0     # Rate of Ca+2 outflux from cell                | s^-1
        Kd = 45e-3          # Kd for ligand binding to channel              | uM
        konA = 0.1          # On-rate for activator binding to channel      | uM^-1 s^-1
        koffA = konA * Kd   # Off-rate for activator leaving channel::A     | s^-1
        konR = 0.1          # On-rate for repressor binding to channel      | uM^-1 s^-1
        koffR = konR * Kd   # Off-rate for repressor leaving channel::R     | s^-1
        konAR = 0.1         # On-rate for repressor binding to channel::A   | uM^-1 s^-1
        koffAR = konAR * Kd # Off-rate for activator leaving channel::A::R  | s^-1
        
        ### Initial Conditions ###
        
        # Activator cell
        ActDNA = 5.0        # Activator ligand DNA                          | Copies
        ActProt = 0.0       # Activator ligand protein (in granules)        | uM
        ActCaC = 100e-3     # Cytoplasmic Ca+2                              | uM
        
        # Repressor cell
        RepDNA = 5.0        # Repressor ligand DNA                          | Copies
        RepProt = 0.0       # Repressor ligand protein (in granules)        | uM
        RepCaC = 100e-3     # Cytoplasmic Ca+2                              | uM
        
        # Ca+2 Channels (assume same for all cells)
        Channel = 100.0     # Ca+2 channels                                 | Copies
        Channel_A = 0.0     # Activated channel                             | Copies
        Channel_R = 0.0     # Repressed channel                             | Copies
        Channel_AR = 0.0    # Dual-bound channel (repression dominates)     | Copies
        
        # Extracellular 
        ActEC = 60e-3       # Activator extracellular ligand                | uM
        RepEC = 0.0         # Repressor extracellular ligand                | uM
        CaEC = 1.8e+3       # Extracellular Ca+2                            | uM
        
        # Collect parameters and initial conditions
        params = [
            ktxn, ktln, kexport, KCa, N, kdp, kdilute, kinflux, koutflux, konA,
            koffA, konR, koffR, konAR, koffAR] 
        init = [
            ActDNA, ActProt, ActCaC,
            RepDNA, RepProt, RepCaC, 
            Channel, Channel_A, Channel_R, Channel_AR,
            ActEC, RepEC, CaEC]
        names = ['Act DNA', 'Act Prot', 'Act Ca_C',
                 'Rep DNA', 'Rep Prot', 'Rep Ca_C',
                 'Channel', 'Channel::A', 'Channel::R', 'Channel::AR',
                 'Act EC', 'Rep EC', 'Ca_EC']
        pnames = ['ktxn', 'ktln', 'kexport', 'KCa', 'N', 'kdp', 'kdilute', 'kinflux', \
                  'koutflux', 'konA', 'koffA', 'konR', 'koffR', 'konAR', 'koffAR']
        return params, init, names, pnames
    
    
    
    
