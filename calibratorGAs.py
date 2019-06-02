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

def Tournament(rPop, numCompetitors, randSeed=0):
	#if randSeed != 0:
		# Initialize random seed
	#	np.random.seed(randSeed)
	# Gives (the index of) 1 winner out of a tournament of numCompetitors number
	# of competitors
	# Create an array with all indicies in rPop: [0,1,2,..., popMax]
	totalPopInd = np.arange(rPop.shape[0])
	# Shuffle it randomly
	np.random.shuffle(totalPopInd)
	# Choose the first numCompetitors
	tournCompetitors = totalPopInd[:numCompetitors]
	# The competitor with the best fitness score is the competitor with the lowest 
	# number, since the population is ordered
	return np.amin(tournCompetitors)






# Survival of the Fittest
def Alg1(rScores, rPop, numOffspring):
	"""
	The easiest algorithm! Send back the top numOffspring number of the best 
	individuals
	"""
	return rPop[:numOffspring]

# Mutation (Asexual Reproduction)
def Alg2(rScores, rPop, numOffspring, valRange, offset=0):
	"""
	We take 10 random species, find the one with the best score, and randomly mutate 
	one of its rotations 10 seperate times to obtain 10 offspring. This whole process 
	is done (numOffspring/10) times.
	"""
	offspring = np.zeros((numOffspring, rPop.shape[1]))
	
	for i in range(int(numOffspring/10.0)):
		# We need to perform a tournament selection to get 1 winner from 4
		parent = rPop[Tournament(rPop, 4)]
		
		for j in range(10):
			#np.random.seed(i+j+seed)
			# We need a location for mutation (node) and a mutation value
			whichNode = np.random.randint(rPop.shape[1])
			# In range [0, rPop.shape[1]-1]
			newVal = np.random.randint(valRange)+offset
			#print("here", whichNode, newVal)
			# Now we put these values into the new offspring
			offspring[i*10+j] = parent
			offspring[i*10+j, whichNode] = newVal
	
	return offspring







# Crossover (Sexual Reproduction)
def Alg3(rScores, rPop, numOffspring, seed):
	"""
	We take two sets of 10 random species, find the best score in each, and randomly 	
	crossover 1 of its genes five seperate times to obtain 10 offspring. Each 
	switch between parents A and B creates two offspring, one made of mostly A and 
	one made with mostly B. This whole This whole process is done (numOffspring/10)
	times. 
	"""
	
	nodesCrossed = int(rPop.shape[1]*0.25)
	offspring = np.zeros((numOffspring, rPop.shape[1])).astype(int)
	
	for i in range(int(numOffspring/10.0)):

		# We need to perform two tournament selections first. 1 winner from 3
		# Gets 2 parents that aren't the same
		parentAind = Tournament(rPop, 2, randSeed=i+seed)
		parentBind = Tournament(rPop, 2, randSeed=i+1+seed)
		counter=i+1
		while parentAind == parentBind:
			#print("while1")
			counter += 1
			parentBind = Tournament(rPop, 2, randSeed=counter+seed)
		parentA = rPop[parentAind]
		parentB = rPop[parentBind]
		for j in range(5):
			# We need unique locations for crossover (node)
			whichNode = np.zeros((nodesCrossed)).astype(int)
			for k in range(nodesCrossed):
				np.random.seed(k+seed)
				whichNode[k] = np.random.randint(rPop.shape[1])

			# This process creates two offspring. First, copy parents over
			offspring[i*10+j*2] = parentA
			offspring[i*10+j*2+1] = parentB

			# Now, switch the node in each offspring
			for k in range(nodesCrossed):
				wNode = whichNode[k]
				offspring[i*10+j*2, wNode] = parentB[wNode]
				offspring[i*10+j*2+1, wNode] = parentA[wNode]
		"""
		if i == 0:
			
			print(offspring[0])
			print(offspring[1])
			print(parentA)
			print(parentB)
		"""
	return offspring



