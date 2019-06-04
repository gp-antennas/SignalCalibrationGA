"""
Written by: 	Suren Gourapura

Date Written: 	06/01/2019

Goal:			Acts as a User Interface to all of the SignalCalibration
				code. Runs the calibratorMain.py, which in turn accesses the
				calibratorFScore.py and calibratorGA.py codes to run the
				whole program. 

Comments:		I wrote this code seperately to simplify the useage of this
				program. Most knobs to adjust the code are shown here.
"""

import numpy as np
import calibratorMain as main

# What save name should the best result have?
saveName = "evolvedValues"

# Maximum number of Generations?
genMax = 200

# Maximum population size?
popMax = 100

# Fitness score type? 1 = dummy, 2 = real
fitType = 1

# Value range?
valRange = 1.

# Mean value? What is the range centered on?
valMean = 0.5


# GENETIC ALGORITHM 1 PARAMETERS
# What number of the next gen is evolved by this algorithm?
# Must be divisible by 10 (e.g. 0, 10, 20, 80, etc.)
Alg1Number = 10


# GENETIC ALGORITHM 2 PARAMETERS
# What number of the next gen is evolved by this algorithm?
# Must be divisible by 10 (e.g. 0, 10, 20, 80, etc.)
Alg2Number = 80
# What fraction of competitors should alg 2's tournament use?
Alg2competitorFrac = 0.25


# GENETIC ALGORITHM 3 PARAMETERS
# What number of the next gen is evolved by this algorithm?
# Must be divisible by 10 (e.g. 0, 10, 20, 80, etc.)
Alg3Number = 10
# What fraction of competitors should alg 3's tournament use?
Alg3competitorFrac = 0.25
# What number of nodes should be crossed over to make offspring?
nodesCrossed = 10




# Fitness score breakdown. Must add to popMax
fitBreakdown = [Alg1Number, Alg2Number, Alg3Number]

main.main(genMax, popMax, fitType, fitBreakdown, \
		valRange, valMean, Alg2competitorFrac,	\
		nodesCrossed, Alg3competitorFrac, saveName)





