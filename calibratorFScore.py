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
import os.path # For checking if indivHistory file exists



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
	elif fitType == -1: # FIX
		scores = np.ones((pop.shape[0]))
	else:
		print("Error: fitType value not 1 or 2")
		return

	# Integrate these new individuals and scores in the indivHistory.csv file
	#UpdateIndivHistory(rScores, rPop, fitType)

	return scores


def FScoreDummy(pop):
	"""
	We grab a premade array of 127 random numbers between 0-1. These 
	values will be our target. To make a new random number array, use 
	the commented code below. To calculate the score, we simply 
	calculate the chi squared between our individual and the goal 
	values (goalVal).
	"""
	goalFile = "data/dummyScore/goalValues.csv"
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



def UpdateIndivHistory(rScores, rPop, fitType):
	# Combine the scores and pop into one matrix. Each row has the score
	# first, then it's associated individual. The rows are ranked best to
	# worst.
	currentData = np.hstack((rScores.reshape((rScores.shape[0],1)), rPop))

	# Choose the file location based on the fitType
	if fitType == 1:
		fileName = "data/dummyScore/indivHistory.csv"
	elif fitType == 2:
		fileName = "data/realScore/indivHistory.csv"

	if os.path.isfile(fileName) is False: # If the file does not exist
		np.savetxt(fileName, currentData, delimiter=',')
	else: # If the file exists
		# We need to insert each element in the correct place in the file
		# Grab the history matrix
		indivHist = np.genfromtxt(fileName, delimiter=',')

		# Create a new combined dataset
		combData = np.zeros((indivHist.shape[0]+currentData.shape[0], \
						indivHist.shape[1]))
		# We need an index for indivHist and for currentData
		hInd, cInd = 0, 0

		for i in range(combData.shape[0]):# For each future entry in combData
			# There are many cases to consider
			# First, check if cInd is out of bounds for currentData
			if cInd + 1 > currentData.shape[0]:
				# If so, just copy over indivHist
				combData[i] = indivHist[hInd]
				hInd += 1
			# Next, check if hInd is out of bounds for indivHist
			elif hInd + 1 > indivHist.shape[0]:
				# If so, just copy over currentData
				combData[i] = currentData[cInd]
				cInd += 1
			# Now, if we have values for both currentData and indivHist,
			# if the current data value is larger
			elif currentData[cInd,0] > indivHist[hInd,0]:
				# Copy over currentData's value
				combData[i] = currentData[cInd]
				cInd += 1
			# Finally, if we have values for both currentData and indivHist,
			# if the indivHist is larger (or equal to)
			else:
				# Copy over indivHist's value
				combData[i] = indivHist[hInd]
				hInd += 1
		# Now, we rewrite indivHistory.csv with the new combData
		np.savetxt(fileName, combData, delimiter=',')
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