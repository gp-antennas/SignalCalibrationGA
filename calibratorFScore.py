"""
Written by: 	Suren Gourapura

Date Written: 	06/01/2019

Goal:			Provide a fitness score for the calibratorMain.py code

Comments:		There will be a main fitness score function called 
				FitnessMain. This will serve to choose which fitness 
				score will be used by the code. There will be a dummy 
				fitness score for now to emulate the real thing.
"""
import numpy as np
import os.path # For checking if indivHistory file exists
import util
import sys

def FitnessTest(pop, fitType, data, amp, phi0):
	"""
	This function is called by calibratorMain.py to choose the fitness
	score type needed. It takes the population and an integer that 
	tells which fitness score the user wants. It returns the ranked 
	scores and ranked population to the calibratorMain.py.
	"""
	if fitType == 1:
		scores = FScoreDummy(pop)
	elif fitType == 2:
		scores = FScoreReal(pop, data, amp, phi0)
	elif fitType == -1: # Used for troubleshooting. Disregard
		scores = np.ones((pop.shape[0]))
	else:
		print("Error: fitType value not 1 or 2")
		return

	# Integrate these new individuals and scores in the 
	# indivHistory.csv file 
	#UpdateIndivHistory(rScores, rPop, fitType)

	return scores

#function to get the data and store it in a matrix. 

#TODO: make nEvents a command line arg

def getData(channel,pop):
        nEvents=50
        

        #the maximum time value.
        maxt=40.-.3125;

        # the evenly sampled times.
        eTimes=np.linspace(0., maxt, 128)
        # zero-crossing period (.5 period over frequency), currently unused
        zCPeriod=.5/1.2;
        #container to hold the scores
        scores = np.zeros((pop.shape[0])) # pop.shape[0] = popMax
        #container to hold the true sample times (from the gA)
        tVec=np.zeros(128);
        #fill a matrix of our data events, would be nice to do this outside of the fitness score loop.
        data=np.zeros((nEvents, 128))        
        phi0=np.zeros(nEvents)
        amp=np.zeros(nEvents)
        for event in range(nEvents):
                #this is the data file. 
                #the first entry is the initial phase
                #the second entry is the amplitude
                #the rest of the entries are the waveform data
                evstr="00"
                chstr=str(channel)
                #print chstr
                if event < 10 :
                        evstr="00"+str(event)
                        
                elif event >= 10 and event < 100:
                        evstr="0"+str(event)
                
                # print evstr
                #infile="/users/PCON0003/osu10643/src/SignalCalibrationGA/data/"+chstr+"withphase"+evstr+".txt"
                infile = "data/"+chstr+"withphase"+evstr+".txt"
                #the max time of the 126th (indexed from 0) sample  
                temp=np.genfromtxt(infile, delimiter="\n")
                phi0[event]=temp[0]

                amp[event]=temp[1]
                trace=temp[2:]
               # phi0[event]=util.getInstPhase(eTimes,trace, 0)
                data[event]=util.normalize(trace)
                
        return data, amp, phi0



def FScoreDummy(pop):
	"""
	We grab a premade array of 126 random numbers between 0-1. These 
	values will be our target. To make a new random number array, use 
	the commented code below. To calculate the score, we simply 
	calculate the chi squared between our individual and the goal 
	values (goalVal).
	"""
	numNodes = pop.shape[1]
	goalFile = "data/dummyScore/goalValues.csv"
	goalVal = np.genfromtxt(goalFile, delimiter=",")[:numNodes]
	
	# Now we create the fitness scores (scores) by calculating the 
	# Chi Squared in a loop
	scores = np.zeros((pop.shape[0])) # pop.shape[0] = popMax

	for i in range(pop.shape[0]):
		# Sum of the chi sq of each delta t with the respective value
		scores[i] = np.sum(((pop[i] - goalVal)**2)/goalVal)

	# To minimize chi squared, we maximize 1/ chi squared
	eps = 10**(-6)
	return 1./(scores + eps)



def FScoreReal(pop, data, amp, phi0):
        #progress indicator
        #print (".")
        #sys.stdout.flush()        
        
        #TODO: 
        #would be good to have an argument for the channel we're calibrating
        #would be nice to make this an argument.
        nEvents=data.shape[0]
        

        #the maximum time value.
        maxt=40.-.3125;

        # the evenly sampled times.
        eTimes=np.linspace(0., maxt, 128)
        # zero-crossing period (.5 period over frequency), currently unused
        zCPeriod=.5/1.2;
        #container to hold the scores
        scores = np.zeros((pop.shape[0])) # pop.shape[0] = popMax
        #container to hold the true sample times (from the gA)
        tVec=np.zeros(128);


        for i in range(scores.size):
                #set the times for entries above 0 to the individual values
                tVec[1:]=pop[i]
  
                #the sample times from the GA
                rTimes=tVec+eTimes

                for event in range(nEvents):
#                        print phi0[event], amp[event]
                        cw=util.sampledCW(1.2, amp[event], rTimes, phi0[event])                        
                        ipScore=np.dot(util.normalize(cw), data[event])
                        scores[i]+=ipScore
        return scores/nEvents





def UpdateIndivHistory(rScores, rPop, fitType):
	# Combine the scores and pop into one matrix. Each row has the 
	# score first, then it's associated individual. The rows are 
	# ranked best to worst.
	currentData = np.hstack((rScores.reshape((rScores.shape[0],1)), \
							rPop))

	# Choose the file location based on the fitType
	if fitType == 1:
		fileName = "data/dummyScore/indivHistory.csv"
	elif fitType == 2:
		fileName = "data/realScore/indivHistory.csv"

	if os.path.isfile(fileName) is False: # If the file does not exist
		np.savetxt(fileName, currentData, delimiter=',')
	else: # If the file exists
		# We need to insert each element in the correct place in file
		# Grab the history matrix
		indivHist = np.genfromtxt(fileName, delimiter=',')

		# Create a new combined dataset
		combData = np.zeros((indivHist.shape[0]+currentData.shape[0], \
						indivHist.shape[1]))
		# We need an index for indivHist and for currentData
		hInd, cInd = 0, 0

		# For each future entry in combData
		for i in range(combData.shape[0]):
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
			# Now if we have values for both currentData and indivHist
			# if the current data value is larger
			elif currentData[cInd,0] > indivHist[hInd,0]:
				# Copy over currentData's value
				combData[i] = currentData[cInd]
				cInd += 1
			# Finally if we have values for currentData and indivHist,
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
generates 126 random numbers from 0 to 1 and stores it in a file with name
goalStr.
"""

"""
import random

strName = "goalString"
goalStr = np.zeros((126))
for i in range(126):
	goalStr[i] = random.random()

print(goalStr)

np.savetxt('data/'+strName+'.csv', goalStr, delimiter=',')

"""
