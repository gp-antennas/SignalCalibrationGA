"""
Written by: 	Suren Gourapura

Date Written: 	06/01/2019

Goal:			Contain the main function called by the calibratorUI.py,
				and runs the whole code. 

Comments:		This code is the heart of the program. We first 
				initialize the population (pop), we calculate each
				individual's fitness score (fScore), then we place them
				into the loop, which evolves the population using genetic
				algorithms and tests their fitness again, before evolving 
				some more. After a fixed number of generations (genMax),
				we display the results.
"""
import numpy as np
import calibratorGA as GA
import calibratorFScore as FScore




def CheckSudoku(problem, sendErrs=False):
	# Given the square to check for, the problem, and the value to look for,
	# Return number of times found. Error if zero
	def CheckSquare(rowStart, colStart, valLoc, prob):
		# Grab the row and column of the value
		rVal, cVal = valLoc
		# Grab the value from the problem matrix
		val = prob[rVal, cVal]
		# Initialize the locatons found matrix (identical to errorLoc)
		locFound = np.zeros((9,9))

		for r in range(rowStart, rowStart+3):
			for c in range(colStart, colStart+3):
				# If the value is equal to val and the locations 
				# aren't the same:
				if equal(prob[r, c], val) and (r != rVal \
					and c != cVal):
					locFound[r, c] = 1
		return locFound
		

	# Create a matrix to hold whether each number issues an error
	errorLoc = np.zeros((9, 9))

	for r1 in range(9):
		for c1 in range(9):
			# Store the value of the problem at this row and column
			val = problem[r1, c1]
			# Check if the value is greater than 0. (we use 0.5 in case
			# val = 0.00000001 or something).
			if val > 0.5:
				# Search all OTHER values in the same column
				for r2 in range(9):
					if equal(problem[r2, c1], val) and r2 != r1:
						errorLoc[r2, c1] = 1

				# Search all OTHER values in the same row
				for c2 in range(9):
					if equal(problem[r1, c2], val) and c2 != c1:
						errorLoc[r1, c2] = 1

				# Search all OTHER values in the same square
				# Find out which square it's in
				topRow = r1 < 3
				midRow = r1 >= 3 and r1 < 6
				botRow = r1 >= 6

				leftCol = c1 < 3
				midCol = c1 >= 3 and c1 < 6
				rightCol = c1 >= 6
				# Create tuple for search location
				valLoc = r1, c1
				
				if leftCol and topRow:
					errorLoc += CheckSquare(0, 0, valLoc, problem)
				if midCol and topRow:
					errorLoc += CheckSquare(0, 3, valLoc, problem)
				if rightCol and topRow:
					errorLoc += CheckSquare(0, 6, valLoc, problem)

				if leftCol and midRow:
					errorLoc += CheckSquare(3, 0, valLoc, problem)
				if midCol and midRow:
					errorLoc += CheckSquare(3, 3, valLoc, problem)
				if rightCol and midRow:
					errorLoc += CheckSquare(3, 6, valLoc, problem)		

				if leftCol and botRow:
					errorLoc += CheckSquare(6, 0, valLoc, problem)
				if midCol and botRow:
					errorLoc += CheckSquare(6, 3, valLoc, problem)
				if rightCol and botRow:
					errorLoc += CheckSquare(6, 6, valLoc, problem)
	""" Now, all the hard work is done. But, places that errored many times may now
	have the value of 2 or more. To prevent this, we go through each value and 
	reassign any values greater than 1 to be 1"""	
	for r1 in range(9):
		for c1 in range(9):
			if errorLoc[r1, c1] > 1.5:
				errorLoc[r1, c1] = 1

	if sendErrs:
		return MatrixInt(np.sum(errorLoc)), MatrixInt(errorLoc)
	else:
		return MatrixInt(np.sum(errorLoc))


def GuessOnErrors(problem, seed=0, cycles=1000):
	print('\nGuessing over Errors')

	# Get the errors
	numErrs, errMatrix = CheckSudoku(problem, sendErrs=True)
	errInds = np.argwhere(errMatrix>0.5)
	errInds = np.concatenate((errInds, np.argwhere(problem < 0.5)))

	# Replace those places with zero
	zeroedProb = MatrixInt(np.array([i for i in problem]))
	for i in range(errInds.shape[0]):
		r, c = errInds[i]
		zeroedProb[r, c] = 0
	
	# Create a guesses matrix
	guesses = MatrixInt(np.ones((9, 9, 9)))
	for i in range(9):
		for j in range(9):
			if zeroedProb[i, j] != 0:	
				guesses[i, j] = np.zeros((9))
	
	# First, check the rows and columns of the provided numbers to eliminate guesses
	guesses = LogicSolve.CheckColAndRow(guesses, zeroedProb)
	# Next, check the square around each provided number to eliminate guesses
	guesses = LogicSolve.CheckSquare(guesses, zeroedProb)
	
	print("With ", np.argwhere(guesses>0.5).shape[0], "possibilities")

	cycle = 0
	FoundAns = False
	while cycle < cycles and FoundAns == False:

		solnGuess = MatrixInt(np.array([i for i in zeroedProb]))

		# For each errored square
		for i in range(errInds.shape[0]):

			# Create the random seed
			np.random.seed(seed+cycles*errInds.shape[0]+i)
			# grab the relevant indicies and guesses
			r, c = errInds[i]
			thisSquareGuesses = guesses[r,c]

			# We are grabbing a list of all possible values in this square
			possibleVals = []
			for j, val in enumerate(thisSquareGuesses):
				if val > 0.5:
					possibleVals.append(j+1)
			
			# Finally, we place one of these values randomly in the square
			randInd = np.random.randint(len(possibleVals))
			solnGuess[r, c] = possibleVals[randInd]
		
		# Now we check if we have solved it
		numErrs = CheckSudoku(solnGuess)
		if numErrs < 0.5:
			FoundAns = True
		cycle += 1

	return FoundAns, solnGuess

def FScore1(pop, problem, numBank, eps=10**(-5)):
	"""
	The fitness score is the inverse of the number of errors each solution provides.
	We construct the solutions using pop and numBank, and test these with CheckSudoku
	to get our fScore array.
	"""
	def Sort(scores, indivs):
		# This function sorts both scores and indivs from greatest to least score
		# Check first to make sure the arrays have the same length
		if scores.shape[0] != indivs.shape[0]:
			print("Error, sorting sizes are not the same.")
			return
		combined = zip(scores, indivs)
		sortedScores = sorted(combined, key=lambda t:t[0], reverse=True)
		#sortedScores = combined.sort(key=lambda t:t[0] ,reverse=True)
		rScores, rIndivs = zip(*sortedScores)
		return np.asarray(rScores), np.asarray(rIndivs)

	# Each individual gets a fitness score
	fScore = np.zeros((pop.shape[0]))

	for indivInd in range(pop.shape[0]): # For each individual
		# Grab the individual
		indiv = pop[indivInd]

		# Initialize the solution bank. I do this nasty way to copy the 
		# numBank because it prevents solnBank == numBank
		solnBank = MatrixInt(np.array([i for i in numBank]))

		# Initialize the solution. Same logic as above
		solnProb = MatrixInt(np.array([i for i in problem]))

		for i in range(indiv.shape[0]): # For each swap command
			# Swap the ith individual with what indiv tells us
			solnBank[i], solnBank[indiv[i]] = solnBank[indiv[i]], solnBank[i]

		# Now we fill our solution with the values in the solution bank
		emptySpots = np.argwhere(problem==0)
		for i in range(emptySpots.shape[0]):
			r, c = emptySpots[i]
			solnProb[r, c] = solnBank[i]

		# Get the number of errors in the solution
		numErrs = CheckSudoku(solnProb)
	
		# Finally, fill in the fitness score
		fScore[indivInd] = 1. / (numErrs + eps)

		# If the score is perfect:
		perfMat = 0
		if numErrs < 0.5:
			perfMat = solnProb
		
	# We now need sort the scores and individuals, and send them back
	rFScores, rPop = Sort(fScore, pop)
	return rFScores, rPop, perfMat


def SolveSudoku1(prob, nPop=50, gens = 1000, randSeed=0, fitBreakdown=[10, 30, 10]):
	eps = eps=10**(-3)
	
	# Initialize random seed
	np.random.seed(randSeed)

	# Convert numbers to integers
	problem = MatrixInt(prob)

	# Print problem
	print("Got this: \n", problem, "\n")

	# Check if the provided Matrix is valid. If the number of errors is != 0:
	if equal(CheckSudoku(problem), 0) is False:
		print("Error: Sudoku is unsolveable\n")
		return


	# INITIALIZE "NUMBER BANK" OF NUMBER TO CHOOSE FROM
	# We first need to record how many of each number the problem has
	numNumbers = np.zeros((10)).astype(int)
	for num in range(10): # For each possible number
		# Record the amount of times that number shows up
		numNumbers[num] = np.argwhere(problem==num).shape[0]

	""" Now, we need to make an array of numbers that has each number (1-9) show
	up as many times as it should to solve the sudoku (Need 9 of each number)."""
	# First, get a list of 1's with 9-numOnes elements
	numberBank = np.ones(9-numNumbers[1]).astype(int)
	for i in range(2,10):
		# Create that number's list and stick it onto the number bank
		nextNumList = MatrixInt(np.ones(9-numNumbers[i])*i)
		numberBank = np.concatenate((numberBank, nextNumList))


	# INITIALIZE DNA FOR SWAPPING

	# The number of locations we need to swap. Also equal to numNumbers[0]
	numLoc = numNumbers[0]

	""" The population holds nPop individuals. For each empty location, the individual
	holds the index of another empty location that it wants to swap with to obtain
	the final solution."""
	pop = np.random.randint(numLoc, size=(nPop, numLoc))
	
	
	
	# BEGIN EVOLUTION
	
	bestScores = np.zeros((gens))

	for gen in range(gens):
		
		# We need to test each of these individuals and obtain fitness scores
		# The function returns them both in ranked order, best to worst
		rankedFScores, rankedPop, b = FScore(pop, problem,  \
							numBank=numberBank, eps=eps)
		
		# If the best one is a perfect solution, print the solution and stop loop
		if equal(perfMat, 0) is False:
			print( "Found! Generation: ", gen)
			print(perfMat)
			return

		if gen%10 == 0:
			print("Gen "+str(gen)+" Scores"+str(1/rankedFScores[:8]-eps))
		# Record the best one
		bestScores[gen] = 1./rankedFScores[0]-eps
		
		# We now begin creating the new population
		newPop1 = GA.Alg1(rankedFScores, rankedPop, fitBreakdown[0])
		newPop2 = GA.Alg2(rankedFScores, rankedPop, fitBreakdown[1], numLoc, \
				offset=0)
		newPop3 = GA.Alg3(rankedFScores, rankedPop, fitBreakdown[2], gen*1000)
		
		newPop = MatrixInt(np.vstack((newPop1, newPop2, newPop3)))
		if EqualMatrices(pop, newPop):
			print("AreSame")
		# Equate the old pop to the new pop to allow the cycle to begin over
		pop = newPop
	
	# Rank the scores one last time
	rankedFScores, rankedPop, perfMat= FScore(pop, problem, numBank=numberBank,\
						 eps=eps)
	
	print("\nFinal Scores: ", 1/rankedFScores[:8]-eps)

	print("Best Score Array: ")
	#for i in bestScores:
	#	print(i)
	
	return

def FScore(pop, problem, eps=10**(-5)):
	"""
	The fitness score is the inverse of the number of errors each solution provides.
	We construct the solutions using pop and numBank, and test these with CheckSudoku
	to get our fScore array.
	"""
	def Sort(scores, indivs):
		# This function sorts both scores and indivs from greatest to least score
		# Check first to make sure the arrays have the same length
		if scores.shape[0] != indivs.shape[0]:
			print("Error, sorting sizes are not the same.")
			return
		combined = zip(scores, indivs)
		sortedScores = sorted(combined, key=lambda t:t[0], reverse=True)
		#sortedScores = combined.sort(key=lambda t:t[0] ,reverse=True)
		rScores, rIndivs = zip(*sortedScores)
		return np.asarray(rScores), np.asarray(rIndivs)

	# Each individual gets a fitness score
	fScore = np.zeros((pop.shape[0]))

	for indivInd in range(pop.shape[0]): # For each individual
		# Grab the individual
		indiv = pop[indivInd]

		# Initialize the solution. I do this nasty way to copy the 
		# numBank because it prevents solnBank == numBank
		solnProb = MatrixInt(np.array([i for i in problem]))

		# Now we fill our solution with the values in the solution bank
		emptySpots = np.argwhere(problem==0)
		for i in range(emptySpots.shape[0]):
			r, c = emptySpots[i]
			solnProb[r, c] = indiv[i]

		# Get the number of errors in the solution
		numErrs = CheckSudoku(solnProb)
	
		# Finally, fill in the fitness score
		fScore[indivInd] = 1. / (numErrs + eps)

		
	# We now need sort the scores and individuals, and send them back
	rFScores, rPop = Sort(fScore, pop)

	# Store the best matrix seperately
	bestSoln = MatrixInt(np.array([i for i in problem]))
	emptySpots = np.argwhere(problem==0)
	for i in range(emptySpots.shape[0]):
		r, c = emptySpots[i]
		bestSoln[r, c] = rPop[0,i]
	
	# If the score is perfect:
	perfect = False
	if 1./rFScores[0] - eps < 0.5:
		perfect = True

	return rFScores, rPop, bestSoln, perfect
	
		



def SolveSudoku(prob, nPop=50, gens = 1000, randSeed=1, fitBreakdown=[10, 30, 10]):
	eps = eps=10**(-3)
	
	# Initialize random seed
	np.random.seed(randSeed)

	# Convert numbers to integers
	problem = MatrixInt(prob)

	# Print problem
	print("Got this: \n", problem, "\n")

	# Check if the provided Matrix is valid. If the number of errors is != 0:
	if equal(CheckSudoku(problem), 0) is False:
		print("Error: Sudoku is unsolveable\n")
		return

 	# INITIALIZE "NUMBER BANK" OF NUMBER TO CHOOSE FROM
	# We first need to record how many of each number the problem has
	numNumbers = np.zeros((10)).astype(int)
	for num in range(10): # For each possible number
		# Record the amount of times that number shows up
		numNumbers[num] = np.argwhere(problem==num).shape[0]


	# INITIALIZE DNA FOR SWAPPING

	# The number of locations we need to swap. Also equal to numNumbers[0]
	numLoc = numNumbers[0]

	""" The population holds nPop individuals. For each empty location, the individual
	holds the number it wants to put in that empty space."""
	pop = np.random.randint(9, size=(nPop, numLoc))+1
	
	
	
	# BEGIN EVOLUTION
	
	bestScores = np.zeros((gens))

	for gen in range(gens):
		
		# We need to test each of these individuals and obtain fitness scores
		# The function returns them both in ranked order, best to worst
		rankedFScores, rankedPop, bestSoln, perfect = FScore(pop, problem, eps=eps)

		# If the best one is a perfect solution, print the solution and stop loop
		if perfect:
			print( "Found! Generation: ", gen)
			PrintMat(bestSoln)
			return

		if gen%10 == 0:
			print("Gen "+str(gen)+" Scores"+str(1/rankedFScores[:8]-eps))
		
		if gen%10 == 0 and 1/rankedFScores[0]-eps < 24:
			FoundAns, solution = GuessOnErrors(bestSoln, seed=gen*1000)
			if FoundAns:
				print( "Found! Generation: ", gen)
				print(solution)
				return
		
		# Record the best one
		bestScores[gen] = 1./rankedFScores[0]-eps

		# We now begin creating the new population
		newPop1 = GA.Alg1(rankedFScores, rankedPop, fitBreakdown[0])
		newPop2 = GA.Alg2(rankedFScores, rankedPop, fitBreakdown[1], 9, \
				offset=1)
		newPop3 = GA.Alg3(rankedFScores, rankedPop, fitBreakdown[2], gen*1000)
		
		newPop = MatrixInt(np.vstack((newPop1, newPop2, newPop3)))
		if EqualMatrices(pop, newPop):
			print("AreSame")
		# Equate the old pop to the new pop to allow the cycle to begin over
		pop = newPop
	
	# Rank the scores one last time
	rankedFScores, rankedPop, bestSoln, perfect = FScore(pop, problem, eps=eps)
	print("\nWas the best final solution perfect? :", perfect)
	print("It was:\n")
	PrintMat(bestSoln)
	print("error matrix looks like:")
	numErrs, errs = CheckSudoku(bestSoln, sendErrs=True)
	print(errs)
	print("It had", numErrs, "number of errors")
	print("\nFinal Scores: ", 1/rankedFScores[:8]-eps)

	print("Best Score Array: ")
	#for i in bestScores:
	#	print(i)
	
	return


def EqualMatrices(A, B):
	check = np.equal(A, B)
	ret = True
	
	check = check.flatten()
	for i in range(check.shape[0]):
		if check[i]==False:
			ret=False
	return ret




# the zeros stand for blank areas
p1 = np.array([	[0., 0., 0., 0., 0., 0., 0.,0.,0.], 
		[0., 0., 0., 0., 0., 3., 0., 8., 5.],
		[0., 0., 1., 0., 2., 0., 0., 0., 0.],
		[0., 0., 0., 5., 0., 7., 0., 0., 0.],
		[0., 0., 4., 0., 0., 0., 1., 0., 0.],
		[0., 9., 0., 0., 0., 0., 0., 0., 0.],
		[5., 0., 0., 0., 0., 0., 0., 7., 3.],
		[0., 0., 2., 0., 1., 0., 0., 0., 0.],
		[0., 0., 0., 0., 4., 0., 0., 0., 9.]] )

empty = np.array([	[0, 0, 0, 0, 0, 0, 0, 0, 0], 
		[0, 0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 0]] )

p3 = np.array([	[8, 7, 6, 9, 0, 0, 0, 0, 0], 
		[0, 1, 0, 0, 0, 6, 0, 0, 0],
		[0, 4, 0, 3, 0, 5, 8, 0, 0],
		[4, 0, 0, 0, 0, 0, 2, 1, 0],
		[0, 9, 0, 5, 0, 0, 0, 0, 0],
		[0, 5, 0, 0, 4, 0, 3, 0, 6],
		[0, 2, 9, 0, 0, 0, 0, 0, 8],
		[0, 0, 4, 6, 9, 0, 1, 7, 3],
		[0, 0, 0, 0, 0, 1, 0, 0, 4]] )


"""

"""
#SolveSudoku(p3)
foundAns, solnGuess = GuessOnErrors(p3, seed=0, cycles=10000)
print(foundAns, solnGuess)







