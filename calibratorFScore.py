"""
Written by: 	Suren Gourapura

Date Written: 	06/01/2019

Goal:			Provide a fitness score for the calibratorMain.py code

Comments:		There will be a main fitness score function called 
				FitnessMain. This will serve to choose which fitness score
				will be used by the code. There will be a dummy fitness 
				score for now to emulate the real thing.
"""
import numpy as np



def Sort(scores, pop):
	"""
	This function sorts both the fitness scores and the individuals
	in a population from greatest score to least score.Check first 
	to make sure the scores and pop arrays have the same length.
	"""
	if scores.shape[0]!=pop.shape[0]:
		print("Error, sorting sizes are not the same.")
		return
	# zip them into one list
	combined = zip(scores, pop)
	# Sort from greatest to least
	sortedScores = sorted(combined, key=lambda t:t[0], reverse=True)
	# unzip them into ranked scores and ranked population
	rScores, rPop = zip(*sortedScores)
	# Return them as numpy arrays (not lists)
	return np.asarray(rScores), np.asarray(rPop)



def FitnessTest(pop, fitType):
	"""
	This function is called by calibratorMain.py to choose the fitness
	score type needed. It takes the population and an integer that tells
	which fitness score the user wants. It returns the ranked scores and
	ranked population to the calibratorMain.py.
	"""
	if fitType == 1:
		scores = FScoreDummy(pop)
	elif fitType == 2:
		scores = FScoreReal(pop)
	else:
		print("Error: fitType value not 1 or 2")
		return
	# Now sort these by greatest to least score
	rScores, rPop = Sort(scores, pop)
	return rScores, rPop


def FScoreDummy(pop):
	"""
	We grab a premade array of 127 random numbers between 0-1. These 
	values will be our target. To make a new random number array, use 
	the commented code below. To calculate the score, we simply 
	calculate the chi squared between our individual and the goal 
	values (goalVal).
	"""
	goalFile = "data/goalValues.csv"
	goalVal = np.genfromtxt(goalFile, delimiter=",")
	
	# Now we create the fitness scores (scores) by calculating the 
	# Chi Squared in a loop
	scores = np.zeros((pop.shape[0])) # pop.shape[0] = popMax

	for i in range(pop.shape[0]):
		# The sum of the chi sq of each delta t with the respective value
		scores[i] = np.sum(((pop[i] - goalVal)**2)/goalVal)

	# To minimize chi squared, we maximize 1/ chi squared
	eps = 10**(-6)
	return 1./(scores + eps)


def FScoreReal(pop):

	return


"""
The below code writes a new goal string for the simple fitness score. It
generates 127 random numbers from 0 to 1 and stores it in a file with name
goalStr.
"""

"""
import random

strName = "goalString"
goalStr = np.zeros((127))
for i in range(127):
	goalStr[i] = random.random()

print(goalStr)

np.savetxt('data/'+strName+'.csv', goalStr, delimiter=',')

"""