"""
Written by: 	Suren Gourapura

Date Written: 	06/01/2019

Goal:			Contain the main function called by the calibratorUI.py,
				and runs the whole code. 

Comments:		This code is the heart of the program. We first 
				initialize the population (pop), we calculate each
				individual's fitness score, then we place them
				into the loop, which evolves the population using genetic
				algorithms and tests their fitness again, before evolving 
				some more. After a fixed number of generations (genMax),
				we display the results.
"""
import numpy as np
import random

import calibratorGA as GA
import calibratorFScore as FScore



def main(genMax=100, popMax=50, fitType=1, fitBreakdown=[10,80,10], \
		valRange=1, meanVal=0.5, Alg2competitorFrac=0.25,	\
		nodesCrossed=4, Alg3competitorFrac=0.25, saveName="evolvedValues"):

	# Print what the user chose
	PrintIC(genMax, popMax, fitType, fitBreakdown, valRange, meanVal, \
		Alg2competitorFrac,	nodesCrossed, Alg3competitorFrac)

	# Check to make sure sum of fitBreakdown = popMax
	if np.sum(np.array(fitBreakdown)) != popMax:
		print("Error: sum of fitness score breakdown does not equal popMax")
		return

 	# INITIALIZE POPULATION
	pop = np.zeros((popMax, 127))

	for indiv in range(popMax): # For each individual
		for node in range(127): # For each node
			pop[indiv,node] = valRange*(random.random() - 0.5) + meanVal

	# CALCULATE FITNESS SCORES
	# These are the ranked fitness scores and ranked population
	rScore, rPop = FScore.FitnessTest(pop, fitType)

	# BEGIN EVOLUTION
	bestScores = np.zeros((genMax))

	for gen in range(genMax): # For each generation

		# A print statement to show the user current progress
		if gen%10 == 0:
			PrintGenData(rScore, rPop, valRange, gen)
		
		# Record the best one
		bestScores[gen] = rScore[0]

		# RUN GENETIC ALGORITHMS
		# We now begin creating the new population
		newPop1 = GA.Alg1(rScore, rPop, fitBreakdown[0])

		newPop2 = GA.Alg2(rScore, rPop, fitBreakdown[1], valRange, \
					meanVal, Alg2competitorFrac)

		newPop3 = GA.Alg3(rScore, rPop, fitBreakdown[2], nodesCrossed,
					Alg3competitorFrac)
		
		newPop = np.vstack((newPop1, newPop2, newPop3))

		# CALCULATE FITNESS SCORES
		rScore, rPop = FScore.FitnessTest(newPop, fitType)
	
	# Record the last score
	bestScores[gen] = rScore[0]
	
	# Print the last generation's data
	print("\nLast Generation")
	PrintGenData(rScore, rPop, valRange, genMax)

	#print("Best Score Array: ", rPop[0])

	# Save the best one with name saveName
	if fitType == 1:
		np.savetxt('results/dummyScore/'+saveName+'_Gen'+str(genMax)+'.csv', rPop[0], \
			delimiter=',')
	elif fitType == 2:
		np.savetxt('results/realScore/'+saveName+'_Gen'+str(genMax)+'.csv', rPop[0], \
			delimiter=',')		
	
	# Plot results
	Plot(bestScores, rPop[0], genMax, saveName)

	return



def Plot(bestScores, bestIndiv, genMax, saveName):
	# Plot the result using matplotlib
	import matplotlib.pyplot as plt
	# Allows for integer ticks in plot's x axis
	from matplotlib.ticker import MaxNLocator

	# Make a list [0, 1, 2, ...] for generations
	genVec = np.arange((bestScores.shape[0]))
	# Make a list [0, 1, 2, ..., 126] for delta t (nodes)
	nodeVec = np.arange((bestIndiv.shape[0]))

	# Grab the goal values vector
	goalFile = "data/dummyScore/goalValues.csv"
	goalVal = np.genfromtxt(goalFile, delimiter=",")
	
	# Create a plot with 2 axes
	fig = plt.figure(figsize=(30,8))
	ax1 = fig.add_subplot(1,2,1)
	ax2 = fig.add_subplot(1,2,2)

	# Plot progress over generations
	ax1.scatter(genVec, bestScores, color='green', marker='o') 
	
	ax1.set_xlabel('Generation', fontsize=18)
	ax1.set_ylabel('Fitness Score (Inverse of Chi Squared)', fontsize=18)
	ax1.set_title('Fitness Scores over the Generations', fontsize=22)
	ax1.xaxis.set_tick_params(labelsize=20)
	ax1.yaxis.set_tick_params(labelsize=20)
	# Force integer ticks
	ax1.xaxis.set_major_locator(MaxNLocator(integer=True)) 


	# Plot goal vector in green
	ax2.plot(nodeVec, goalVal, c='g') # Plot original data
	# Plot best result vector in red dotted
	ax2.plot(nodeVec, bestIndiv, c='r', linestyle='--')
	ax2.set_xlabel('Node Number', fontsize=18)
	ax2.set_ylabel('Value at Node', fontsize=18)
	ax2.set_title('Best Solution vs. Goal', fontsize=22)
	ax2.xaxis.set_tick_params(labelsize=20)
	ax2.yaxis.set_tick_params(labelsize=20)
	# Force integer ticks
	ax2.xaxis.set_major_locator(MaxNLocator(integer=True)) 
	
	plt.savefig('results/'+saveName+'_Gen'+str(genMax)+'Plot.png')
	plt.show()

def PrintIC(genMax, popMax, fitType, fitBreakdown, valRange, meanVal, \
		Alg2competitorFrac,	nodesCrossed, Alg3competitorFrac):
	print("Your Chosen Evolution Parameters:\n")
	print("Number of generations: "+str(genMax))
	print("Number of individuals in a population:"+str(popMax)+"\n")

	if fitType == 1:
		print("You have chosen the dummy fitness score")
	elif fitType == 2:
		print("You have chosen the real fitness score")

	print("Data values are centered on "+str(meanVal)+" and have a range of "\
		+str(valRange)+" (so +/- "+str(valRange/2.)+")\n")

	print("For Algorithm 1 (Survival of the Fittest), you have chosen:")
	print(str(fitBreakdown[0])+" Individals\n")

	print("For Algorithm 2 (Mutation), you have chosen:")
	print(str(fitBreakdown[1])+" Individals")
	print("A tournament competitors fraction of: "+str(Alg2competitorFrac)+"\n")

	print("For Algorithm 3 (Crossover), you have chosen:")
	print(str(fitBreakdown[2])+" Individals")
	print("A tournament competitors fraction of: "+str(Alg3competitorFrac))
	print("Number of nodes crossed in offspring are: "+str(nodesCrossed)+"\n")

def PrintGenData(rScore, rPop, valRange, gen):
	# Calculate the diversity
	div = CalcDiversity(rPop, valRange)
	# Print the generation's data
	print("Gen: "+str(gen)+f", Diversity: {div:.1f}%, Top 4 Scores: "+ \
		str(rScore[:4]))
	return

def CalcDiversity(indivs, valRange):
	"""
	To show how diverse each generation is, we calculate a percent representing
	this diversity. it is normalized so that 100% is the result of a ranomly
	populated population. 
	"""
	diversity = 0
	for nodeInd in range(indivs.shape[1]):
		diversity += np.std(indivs[:,nodeInd])

	# The number 0.36442 was found by testing diversity/valRange for random 
	# populations and seeing what we got.
	return diversity/(valRange*0.36442)
