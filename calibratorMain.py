"""
Written by:     Suren Gourapura

Date Written:     06/01/2019

Goal:            Contain the main function called by the calibratorUI.py,
                and runs the whole code. 

Comments:        This code is the heart of the program. We first 
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



def main(genMax=100, popMax=50, numNodes=126, fitType=1, fitBreakdown=[10,80,10],  valRange=1, meanVal=0.5, Alg2competitorFrac=0.25, nodesCrossed=4,   Alg3competitorFrac=0.25, epsPercent=10**(-3), useAlg4 =True, saveName="evolvedValues", channel=0):

    # Print what the user chose
    PrintIC(genMax, popMax, fitType, fitBreakdown, valRange, meanVal,       Alg2competitorFrac,    nodesCrossed, Alg3competitorFrac, epsPercent,         useAlg4)

    # Check to make sure sum of fitBreakdown = popMax
    if np.sum(np.array(fitBreakdown)) != popMax:
        print("Error: sum of fitness score breakdown does not equal popMax")
        return

    # INITIALIZE POPULATION
    pop = np.zeros((popMax, numNodes))
    #initialize the fitness data
    data,amp,phi0=FScore.getData(channel, pop)


    for indiv in range(popMax): # For each individual
        for node in range(numNodes): # For each node
            pop[indiv,node] = valRange*(random.random() - 0.5) + meanVal

    # CALCULATE FITNESS SCORES
    # These are the fitness scores from pur population
    scores = FScore.FitnessTest(pop, fitType, data, amp, phi0)
    # We now rank them
    rScores, rPop = Sort(scores, pop)

    # BEGIN EVOLUTION
    bestScores = np.zeros((genMax))

    for gen in range(genMax): # For each generation

        # A print statement to show the user current progress
        if gen%10 == 0:
            PrintGenData(rScores, rPop, valRange, gen)
        
        # Record the best score
        bestScores[gen] = rScores[0]

        # RUN GENETIC ALGORITHMS
        # We now begin creating the new population
        
        newPop1 = GA.Alg1(rScores, rPop, fitBreakdown[0])
        
        newPop2 = GA.Alg2(rScores, rPop, fitBreakdown[1], valRange, 
                    meanVal, Alg2competitorFrac)
        
        newPop3 = GA.Alg3(rScores, rPop, fitBreakdown[2], nodesCrossed,
                    Alg3competitorFrac)
    
        # Combine the results of Alg 2 and Alg 3
        newPop23 = np.vstack((newPop2, newPop3))

        # CALCULATE FITNESS SCORES
        # The scores for Alg1 are just those from the previous gen
        newScores1 = rScores[:fitBreakdown[0]]
        # Calculate the scores for the result of Alg 2 and 3
        newScores23 = FScore.FitnessTest(newPop23, fitType, data, amp, phi0)

        # Stitch the population and scores back together
        newPop = np.vstack((newPop1, newPop23))
        newScores = np.concatenate((newScores1, newScores23))

        # If we are using Algorithm 4:
        if useAlg4:
            # Remove unnecessary duplicate individuals
            divScores, divPop = GA.Alg4(newScores, newPop, valRange, meanVal, fitType, data, amp, phi0, epsPercent=epsPercent)
        else:
            # Just copy it over
            divScores, divPop = newScores, newPop

        # Now finally sort these by greatest to least score
        rScores, rPop = Sort(divScores, divPop)
    
    # Record the last score
    bestScores[gen] = rScores[0]
    
    # Print the last generation's data
    print("Last Generation")
    PrintGenData(rScores, rPop, valRange, genMax)

    #print("Best Score Array: ", rPop[0])

    # Save the best one with name saveName
    if fitType == 1:
        np.savetxt('results/dummyScore/'+saveName+'_ch'+str(channel)+'_Gen'+str(genMax)+'.csv', rPop[0], 
            delimiter=',')
    elif fitType == 2:
        np.savetxt('results/realScore/'+saveName+'_ch'+str(channel)+'_Gen'+str(genMax)+'.csv', rPop[0], 
            delimiter=',')        
    
    # Plot results
    Plot(bestScores, rPop[0], genMax, saveName) # FIX

    return rScores[0]

def Plot(bestScores, bestIndiv, genMax, saveName):
	# Plot the result using matplotlib
	import matplotlib.pyplot as plt
	# Allows for integer ticks in plot's x axis
	from matplotlib.ticker import MaxNLocator

	# Make a list [0, 1, 2, ...] for generations
	genVec = np.arange((bestScores.shape[0]))
	# Make a list [0, 1, 2, ..., 126] for delta t (nodes)
	nodeVec = np.arange((bestIndiv.shape[0]))

	
	# Create a plot with 2 axes
	fig = plt.figure(figsize=(30,8))
	ax1 = fig.add_subplot(1,2,1)
	ax2 = fig.add_subplot(1,2,2)

	# Plot progress over generations
	ax1.scatter(genVec, bestScores, color='green', marker='o') 
	
	ax1.set_xlabel('Generation', fontsize=18)
	ax1.set_ylabel('Fitness Score (Normalized inner product)', fontsize=18)
	ax1.set_title('Fitness Scores over the Generations', fontsize=22)
	ax1.xaxis.set_tick_params(labelsize=20)
	ax1.yaxis.set_tick_params(labelsize=20)
	# Force integer ticks
	ax1.xaxis.set_major_locator(MaxNLocator(integer=True)) 


	# Plot goal vector in green
	print(nodeVec.shape)

#	ax2.plot(nodeVec, goalVal, c='g') # Plot original data
	# Plot best result vector in red dotted
	ax2.plot(nodeVec, bestIndiv, c='r')#, linestyle='--')
	ax2.set_xlabel('Sample Number', fontsize=18)
	ax2.set_ylabel(r'sample $\delta t$ (ns)', fontsize=18)
	ax2.set_title('Best Solution', fontsize=22)
	ax2.xaxis.set_tick_params(labelsize=20)
	ax2.yaxis.set_tick_params(labelsize=20)
	# Force integer ticks
	ax2.xaxis.set_major_locator(MaxNLocator(integer=True)) 
	
	plt.savefig('data/'+saveName+'_Gen'+str(genMax)+'Plot.png')
	plt.show()


# def Plot(bestScores, bestIndiv, genMax, saveName):
#     # Plot the result using matplotlib
#     import matplotlib.pyplot as plt
#     # Allows for integer ticks in plot's x axis
#     from matplotlib.ticker import MaxNLocator

#     # Make a list [0, 1, 2, ...] for generations
#     genVec = np.arange((bestScores.shape[0]))
#     # Make a list [0, 1, 2, ..., 125] for delta t (nodes)
#     nodeVec = np.arange((bestIndiv.shape[0]))

#     # Grab the goal values vector

#     goalFile = "data/dummyScore/goalValues.csv"
#     goalVal = np.genfromtxt(goalFile, delimiter=",")[:bestIndiv.shape[0]]
    
#     # Create a plot with 2 axes
#     fig = plt.figure(figsize=(30,8))
#     ax1 = fig.add_subplot(1,2,1)
#     ax2 = fig.add_subplot(1,2,2)

#     # Plot progress over generations
#     ax1.scatter(genVec, bestScores, color='green', marker='o') 
    
#     ax1.set_xlabel('Generation', fontsize=18)
#     ax1.set_ylabel('Fitness Score (Inverse of Chi Squared)', fontsize=18)
#     ax1.set_title('Fitness Scores over the Generations', fontsize=22)
#     ax1.xaxis.set_tick_params(labelsize=20)
#     ax1.yaxis.set_tick_params(labelsize=20)
#     # Force integer ticks
#     ax1.xaxis.set_major_locator(MaxNLocator(integer=True)) 


#     # Plot goal vector in green
#     ax2.plot(nodeVec, goalVal, c='g') # Plot original data
#     # Plot best result vector in red dotted
#     ax2.plot(nodeVec, bestIndiv, c='r', linestyle='--')
#     ax2.set_xlabel('Node Number', fontsize=18)
#     ax2.set_ylabel('Value at Node', fontsize=18)
#     ax2.set_title('Best Solution vs. Goal', fontsize=22)
#     ax2.xaxis.set_tick_params(labelsize=20)
#     ax2.yaxis.set_tick_params(labelsize=20)
#     # Force integer ticks
#     ax2.xaxis.set_major_locator(MaxNLocator(integer=True)) 
    
#     plt.savefig('results/'+saveName+'_Gen'+str(genMax)+'Plot.png')
#     plt.show()

def PrintIC(genMax, popMax, fitType, fitBreakdown, valRange, meanVal, 
        Alg2competitorFrac,    nodesCrossed, Alg3competitorFrac, epsPercent, 
        useAlg4):
    print("Your Chosen Evolution Parameters:n")
    print("Number of generations: "+str(genMax))
    print("Number of individuals in a population:"+str(popMax)+"n")

    if fitType == 1:
        print("You have chosen the dummy fitness score")
    elif fitType == 2:
        print("You have chosen the real fitness score")

    print("Data values are centered on "+str(meanVal)+" and have a range of "
        +str(valRange)+" (so +/- "+str(valRange/2.)+")n")

    print("For Algorithm 1 (Survival of the Fittest), you have chosen:")
    print(str(fitBreakdown[0])+" Individalsn")

    print("For Algorithm 2 (Mutation), you have chosen:")
    print(str(fitBreakdown[1])+" Individals")
    print("A tournament competitors fraction of: "+str(Alg2competitorFrac)+"n")

    print("For Algorithm 3 (Crossover), you have chosen:")
    print(str(fitBreakdown[2])+" Individals")
    print("A tournament competitors fraction of: "+str(Alg3competitorFrac))
    print("Number of nodes crossed in offspring are: "+str(nodesCrossed)+"n")

    print("For Algorithm 4 (Diversity), you have chosen:")
    print("The fourth algorithm is running: "+str(useAlg4))
    print(f"Individuals must be {epsPercent:.3f}% differentn")

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

def PrintGenData(rScores, rPop, valRange, gen):
    # Calculate the diversity
    div = CalcDiversity(rPop, valRange)
    # Print the generation's data
    print("Gen: "+str(gen)+f", Diversity: {div:.1f}%, Top 4 Scores: "+ 
        str(rScores[:4]))
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
