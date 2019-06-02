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
			print("Gen: "+str(gen)+" Top 4 Scores: "+str(rScore[:4]))
		
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
	
	print("Last Gen: "+str(genMax)+" Top 8 Scores: "+str(rScore[:8]))

	print("Best Score Array: ", rPop[0])

	# Save the best one with name saveName
	np.savetxt('data/'+saveName+'_Gen'+str(genMax)+'.csv', rPop[0], \
		delimiter=',')
	
	# Plot results
	plot(bestScores, rPop[0], genMax, saveName)

	return



def plot(bestScores, bestIndiv, genMax, saveName):
	# Plot the result using matplotlib
	import matplotlib.pyplot as plt
	# Allows for integer ticks in plot's x axis
	from matplotlib.ticker import MaxNLocator

	# Make a list [0, 1, 2, ...] for generations
	genVec = np.arange((bestScores.shape[0]))
	# Make a list [0, 1, 2, ..., 126] for delta t (nodes)
	nodeVec = np.arange((bestIndiv.shape[0]))

	# Grab the goal values vector
	goalFile = "data/goalValues.csv"
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
	print(nodeVec.shape)
	print(goalVal.shape)
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
	
	plt.savefig('data/'+saveName+'_Gen'+str(genMax)+'Plot.png')
	plt.show()

