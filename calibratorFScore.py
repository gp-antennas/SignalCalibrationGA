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
import default as util
import sys

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
                infile="/users/PCON0003/osu10643/src/SignalCalibrationGA/data/"+chstr+"withphase"+evstr+".txt"
                #the max time of the 126th (indexed from 0) sample  
                temp=np.genfromtxt(infile, delimiter="\n")
                phi0[event]=temp[0]

                amp[event]=temp[1]
                trace=temp[2:]
               # phi0[event]=util.getInstPhase(eTimes,trace, 0)
                data[event]=util.normalize(trace)
        return data, amp, phi0


def FitnessTest(pop, fitType, data, amp, phi0):
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
	elif fitType == 3:
                scores = FScoreDev(pop, data, amp, phi0)
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
        #currently test with just one event file. we have 200 and can loop them for better fitness score. 
        print "."
        sys.stdout.flush()        
        infile="/users/PCON0003/osu10643/src/SignalCalibrationGA/data/049.txt"
        #the max time of the 126th (indexed from 0) sample
        maxt=40.-.3125;
        #the sample rate we want for interpolation
        GSs=30.;
        trace=np.genfromtxt(infile, delimiter="\n")
        # the evenly sampled times.
        times=np.linspace(0., maxt, 128)
        # zero-crossing period (.5 period over frequency)
        zCPeriod=.5/1.2;
        scores = np.zeros((pop.shape[0])) # pop.shape[0] = popMax
        tVec=np.zeros(128);
        #print tVec[1:].shape
        #print pop[0].shape
        for i in range(scores.size):
                tVec[1:]=pop[i]
               # print tVec[0], tVec[1], pop[i][0]#tVec[0]=0.
                #the sample times from the GA
                rTimes=tVec+times
                #take these and evenly sample them
                evenSamp=np.interp(times, rTimes, trace);
                #sinc interpolate to get a nice smooth wave.
                sincx, sincy=util.sincInterpolateFast(times, evenSamp, GSs)
                #get the amplitude and phase of this wave               
                amp=util.rms(sincy)*np.sqrt(2)
                phi0= util.getInstPhase(sincx, sincy, 0.)
                #cwx=np.zeros(sincy.size);
                #cwy=np.zeros(sincy.size);
                # if sincy[1]<sincy[0]:
                #         phi0=np.pi-phi0
                cwx, cwy=util.makeCW(1.2, amp, 0., maxt, GSs, phi0)
                #                print cwy.size, sincy.size
                #add condition for consistent sampling
                zC=util.getZeroCross(sincx, sincy)
                zCScore=.01/util.rms(zC-zCPeriod)
                scores[i]=zCScore*np.dot(util.normalize(cwy), util.normalize(sincy))
        return scores


def FScoreDev(pop, data, amp, phi0):
        #progress indicator
        print "."
        sys.stdout.flush()        
        
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
#        fill a matrix of our data events, would be nice to do this outside of the fitness score loop.
        # dataL=np.zeros((nEvents, 128))        
        # phi0=np.zeros(nEvents)
        # amp=np.zeros(nEvents)
        # for event in range(nEvents):
        #         #this is the data file. 
        #         #the first entry is the initial phase
        #         #the second entry is the amplitude
        #         #the rest of the entries are the waveform data
        #         evstr="00"
        #         if event < 10 :
        #                 evstr="00"+str(event)
                        
        #         elif event >= 10 and event < 100:
        #                 evstr="0"+str(event)
                
        #         # print evstr
        #         infile="/users/PCON0003/osu10643/src/SignalCalibrationGA/data/withphase"+evstr+".txt"
        #         #the max time of the 126th (indexed from 0) sample  
        #         temp=np.genfromtxt(infile, delimiter="\n")
        #         phi0[event]=temp[0]

        #         amp[event]=temp[1]
        #         trace=temp[2:]
        #        # phi0[event]=util.getInstPhase(eTimes,trace, 0)
        #         dataL[event]=util.normalize(trace)



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
