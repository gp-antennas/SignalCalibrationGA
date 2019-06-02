"""
Written by: 	Suren Gourapura

Date Written: 	06/01/2019

Goal:			Provide the Genetic Algorithms (GAs) for the 
				calibratorMain.py code

Comments:		There are three genetic algorithms: Alg1, Alg2, and Alg3.
				In addition, we have the selection method called Tournament
				to help in choosing parents for Alg2 and Alg3.

"""
import numpy as np
import random

def Tournament(rPop, numCompetitors):
	"""
	Gives (the index of) 1 winner out of a tournament of numCompetitors
	number of competitors. First, create an array with all indicies in 
	rPop: [0,1,2,..., popMax].
	"""
	totalPopInd = np.arange(rPop.shape[0])
	# Shuffle it randomly
	random.shuffle(totalPopInd)
	# Choose the first numCompetitors
	tournCompetitors = totalPopInd[:numCompetitors]
	# The competitor with the best fitness score is the competitor with 
	# the lowest number, since the population is ordered
	return np.amin(tournCompetitors)



# Survival of the Fittest
def Alg1(rScores, rPop, numOffspring):
	"""
	The easiest algorithm! Send back the top numOffspring number of the
	best individuals
	"""
	return rPop[:numOffspring]



# Mutation (Asexual Reproduction)
def Alg2(rScores, rPop, numOffspring, valRange, meanVal, \
		competitorFrac=0.25):
	"""
	Takes the ranked scores and population, the desired number of 
	offspring (must be divisible by 10), the range of values that an 
	individual's element can take, what fraction of the population should
	be used to tournament select the parents, and what the mean value is.

	We take numOffspring/10 random species, find the one with the best
	score, and randomly mutate one of its elements 10 seperate times 
	to obtain 10 offspring. This whole process is done (numOffspring/10)
	times.
	"""
	offspring = np.zeros((numOffspring, rPop.shape[1]))
	# Calculate the number of competitors
	numCompetitors = int(competitorFrac*rPop.shape[0])
	
	for i in range(int(numOffspring/10.0)):
		# We need to perform a tournament selection to get 1 winner from 4
		parent = rPop[Tournament(rPop, numCompetitors)]
		
		for j in range(10):
			# We need a location for mutation (node) and a mutation value
			# The node can be any value in the range [0, 126]
			whichNode = random.randint(0, rPop.shape[1]-1)
			# Now we choose a new random value for the node
			newVal = valRange*(random.random() - 0.5) + meanVal
			# Now we put these values into the new offspring
			offspring[i*10+j] = parent
			offspring[i*10+j, whichNode] = newVal
	
	return offspring



# Crossover (Sexual Reproduction)
def Alg3(rScores, rPop, numOffspring, nodesCrossed, competitorFrac=0.25):
	"""
	We take two sets of numCompetitor random species, find the best score
	in each set, and randomly crossover nodesCrossed number of its genes 
	five seperate times to obtain 10 offspring. Each switch between 
	parents A and B creates two offspring, one made of mostly A and 
	one made with mostly B. This whole This whole process is done 
	(numOffspring/10) times. 
	"""
	offspring = np.zeros((numOffspring, rPop.shape[1]))
	# Calculate the number of competitors
	numCompetitors = int(competitorFrac*rPop.shape[0])
	
	for i in range(int(numOffspring/10.0)):

		# We need to perform two tournament selections first. 
		parentAind = Tournament(rPop, numCompetitors)
		parentBind = Tournament(rPop, numCompetitors)

		# Gets 2 parents that aren't the same
		while parentAind == parentBind:
			#print("while loop looped")
			parentBind = Tournament(rPop, numCompetitors)

		# Now that you have 2 distinct indicies, get those parents
		parentA = rPop[parentAind]
		parentB = rPop[parentBind]

		# For each of 5 crossovers...
		for j in range(5):
			# We need unique locations for crossover (nodes)
			whichNode = np.zeros((nodesCrossed)).astype(int)
			for k in range(nodesCrossed):
				# The node can be any value in the range [0, 126]
				whichNode[k] = random.randint(0, rPop.shape[1]-1)

			# This process creates two offspring. First, copy parents over
			offspring[i*10+j*2] = parentA
			offspring[i*10+j*2+1] = parentB

			# Now, switch the node in each offspring fom the other parent
			for k in range(nodesCrossed):
				wNode = whichNode[k]

				offspring[i*10+j*2, wNode] = parentB[wNode]
				offspring[i*10+j*2+1, wNode] = parentA[wNode]
	
	return offspring



