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

import calibratorFScore as FScore

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
			# The node can be any value in the range [0, 125]
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
				# The node can be any value in the range [0, 125]
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



def Alg4(scores, pop, valRange, meanVal, fitType, epsPercent=10**(-3)):
	"""
	We don't want duplicates of individuals in our population, nor even almost
	duplicates. So we scan through the NEW population and replace duplicates with 
	a fresh random individual to preserve diversity.
	"""

	def AreSimilar(indivA, indivB):
		# Takes 2 individuals and returns boolean True if they are similar
		# on EVERY node
		similar = True

		for k in range(1, indivA.shape[0]): # For each node (not the fScore)
			# If node A and node B's abs difference is sufficiently large
			if np.abs(indivA[k]-indivB[k]) > epsPercent*valRange/100.:
				# They are not similar
				similar = False

		return similar

	# First, we glue the scores and population to keep them together
	# This command is made complicated because we need to make rScores a 
	# 2d array first using reshape. Then we stitch them together
	popData = np.hstack((scores.reshape((scores.shape[0],1)), pop))

	# Create a blank list that'll hold the index of genetically diverse indivs
	divIndivs = []
	# Put the first element in
	divIndivs.append(popData[0])

	# Count the total number of duplicates
	numDuplicates = 0

	for i in range(1, popData.shape[0]): # For each individual except first

		# We need to count which divIndivs indiv we are checking
		j = 0 # The divIndivs index
		# We also need to go until we find only 1 duplicate
		noithDuplicate = True

		# Scan through the diverse indivs until we find a duplicate
		while j < len(divIndivs) and noithDuplicate: 
			
			# If individual i is not similar to individual j
			if AreSimilar(divIndivs[j], popData[i]) is False:
				# Add it to the list of diverse individuals
				divIndivs.append(popData[i])
				# Flip the flag saying we did find a duplicate
				noithDuplicate = False
			else:
				numDuplicates += 1
				#print("\nfound Duplicate\n") # FIX
			j += 1

	# If there are actually duplicates
	if numDuplicates > .5:
		# Make these diverse individuals a numpy array
		divPopData = np.array(divIndivs)
		# Reseperate fitness score from individuals
		divScores = divPopData[:,0].flatten()
		divPop = divPopData[:,1:]

		# Now we need to add random individuals for the ones we lost
		numRandIndivs = scores.shape[0]-divScores.shape[0]
		randPop = np.zeros((numRandIndivs, pop.shape[1]))

		# Populate the random individuals
		for indiv in range(numRandIndivs): # For each individual
			for node in range(pop.shape[1]): # For each node
				randPop[indiv,node] = valRange*(random.random() - 0.5) + meanVal
		# Calculate their scores
		randScores = FScore.FitnessTest(randPop, fitType)

		# Finally, combine them with the original diverse individuals and send the
		# new population back

		returnScores = np.concatenate((divScores, randScores))
		returnPop = np.vstack((divPop, randPop))

	# If there are no duplicates
	else:
		returnScores = scores
		returnPop = pop
	#print(f"Found {numDuplicates:f}")
	return returnScores, returnPop

'''
scores = np.ones((3))

pop = np.array([[1,2],[1.500001,2],[5,6]])

retScores, retPop = Alg4(scores, pop, valRange=10, meanVal=10, fitType=-1, \
						epsPercent=5)

print(retScores)
print(retPop)
'''