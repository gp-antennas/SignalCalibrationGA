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
import sys

# What save name should the best result have?
saveName = "evolvedVDat"

# Maximum number of Generations?
genMax = 400

# Maximum population size?
popMax = 100

# Fitness score type? 1 = dummy, 2 = real, 3=dev
fitType = 3

# Value range?
valRange = float(sys.argv[2])#.7

# Mean value? What is the range centered on?
meanVal = 0.

#channel
channel=int(sys.argv[1])


# GENETIC ALGORITHM 1 PARAMETERS
# What percent of the next gen is evolved by this algorithm?
Alg1Percent = 20


# GENETIC ALGORITHM 2 PARAMETERS
# What percent of the next gen is evolved by this algorithm?
Alg2Percent = 60
# What fraction of competitors should alg 2's tournament use?
Alg2competitorFrac = 0.25


# GENETIC ALGORITHM 3 PARAMETERS
# What percent of the next gen is evolved by this algorithm?
Alg3Percent = 20
# What fraction of competitors should alg 3's tournament use?
Alg3competitorFrac = 0.25
# What number of nodes should be crossed over to make offspring?
nodesCrossed = 10




# Fitness score breakdown. Must add to popMax
fitBreakdown = [Alg1Percent, Alg2Percent, Alg3Percent]

main.main(genMax, popMax, fitType, fitBreakdown, \
		valRange, meanVal, Alg2competitorFrac,	\
		nodesCrossed, Alg3competitorFrac, saveName, channel)





