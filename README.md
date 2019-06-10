# SignalCalibrationGA

Hello! This the readme for the SURF Signal Calibration Program

OVERVIEW:
	
    This program is used to find the best calibration for the sine
    wave signals recorded at SLAC by the SURF. To run, use the
    following command:
    python3 calibratorUI.py [which channel?] [valRange value?]

CODE:

    calibratorUI.py -
        The user interface for the whole program. Contains most 
        (if not all) variables to play with the code. Most variables
        pertain to the GA. The main function of the code is run at 
        the bottom.

	calibratorMain.py -
		Contains the backbone of the code. Takes all variables from
		calibratorUI and accesses the GA and FScore. Starts by 
		creating a population, calculating their scores, and then 
		starts looping the evolving and score calculation procedure 
		for the specified number of generations. Can print 
		statements every 10 generations, save the best score, and 
		plot the results.

	calibratorGA.py -
		Contains the genetic algorithms and associated functions for 
		the evolutionary process. Each GA may have many inputs, but 
		Algorithms 1-3 take at least the ranked population and the 
		number of competitors it is responsible to output, and outputs
		the created offspring. Algorithm 5 is applied to the whole 
		population, and it returns the population without duplicates.
		The Tournament function tournament selects one individual from
		the ranked population using a selected number of competitors.

	calibratorFScore.py - 
		[Add stuff here]

	util.py -
		Holds all supporting functions for the real fitness score in 
		calibratorFScore.py.

FOLDERS:
	
	data -
		Contains event files from different SURF channels. Also stores
		any extrraneous data we need for the fitness scores

	OldCode -
		Contains the old GA, which only had the first three 
		algorithms. The file addresses are set up ready to run from
		there.

	results -
		Holds the results from any runs we perform. All important runs
		should be saved in their own folder.


AUTHORS:

	The genetic algorithms were written by Suren Gourapura, the 
	fitness score was written by Steven Prohira, and the team of Eliot
	Ferstl, Alex Patton, and Scott Janse tuned the variables to 
	maximize performance.
