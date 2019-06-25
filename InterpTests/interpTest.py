"""
Written by: 	Suren Gourapura

Date Written: 	06/25/2019

Goal:			Test effectiveness of fourier series interpolation
				using real data

Comments:		To change which data we are using, see getData(). To 
				increase precision, increase disc. To increase number 
				of coefficients used, increase fSeriesNMax. To 
				increase left plot's range, increase leftXLim.
"""
import numpy as np
import time


# Function written by Steven. Samples a sine wave at given frequency, 
# amplitude and period at an array of times (or one time)
def sampledCW(freq, amp, times, phase):
	values=amp*np.sin(2.*np.pi*freq*times +phase)
	return values


# Currently, I am using a channel 5 GA run with ~0.991 fitness score 
def getData():
	# Grab the corrected delta T values
	deltaTs = np.genfromtxt("/home/suren/Github/SignalCalibrationGA"+\
				"/results/realScore/forPres_ch5_Gen500.csv")
	# Create a list of the nominal times
	gaTimes = np.arange(128)*40/128
	# Add the GA calibrations to it
	gaTimes[1:] = gaTimes[1:]+deltaTs

	# Grab the y values of one of channel 5 events
	yData = np.genfromtxt("/home/suren/Github/SignalCalibrationGA/"+\
				"data/5withphase000.txt")
	# Seperate data into phase, amp, and gaYVals
	phase = yData[0]
	amp = yData[1]
	gaYVals = yData[2:]

	# Put this stuff into a tuple called gaData
	gaData = gaTimes, gaYVals

	return phase, amp, gaData


# Create the CW that matches the data with num precision
def getCW(T, freq, amp, phase, num=1000):
	# Populate the times
	sinTimes = np.linspace(0, T, num=num)
	# Sample the sine wave with the requested parameters
	sinYVals = sampledCW(freq, amp, sinTimes, phase)
	# Return sinData, which has the x and y to plot the CW
	return sinTimes, sinYVals


# Calculate the m'th fourier coefficient of the data
def CalcFourierCoeffs(m, T, funcData):
	# Seperate x and y values
	xVals, func = funcData
	# Initialize the coefficients as 0
	a_m = 0
	b_m = 0
	# Calculate the integral's value at each t value
	for i, t in enumerate(xVals[:-2]):
		# Dynamically adjust dt to the gap in front of the integral
		# Note that due to "i+1", we cannot integrate over last value
		# So enumerate only goes to xVals[:-2], last but 1
		dt = xVals[i+1] - t
		a_m += func[i]*np.cos(2*np.pi*m*t/T)*dt
		b_m += func[i]*np.sin(2*np.pi*m*t/T)*dt
	# If we are on m=0, we don't multiply by 2 and b_m=0
	if int(round(m)) == 0:
		return 0, a_m/T
	# Otherwise, we multiply by 2
	else:
		return 2*b_m/T, 2*a_m/T


# Use the fourier coefficients to construct the interpolated data
# Can either request evenly sampled data using num datapoints, or 
# can pass own xVals array to get an unevenly sampled interpolation
def FourierSeries(sinCoeffs, cosCoeffs, T, num=0, xVals=0):
	# If an xVals value was not given by user for evenly sampled data
	if np.sum(xVals) < 10e-8:
		# If a num value was also not given by the user
		if num == 0:
			print("Error, num and xVals in FourierSeries() are zero")
			return
		# Otherwise, populate the times
		xVals = np.linspace(0, T, num=num)
	# Add the a_0 coefficient
	sum = cosCoeffs[0]
	# Add the remaining coefficients
	for m in range(1, sinCoeffs.shape[0]):
		sum += sinCoeffs[m]*np.sin(2*np.pi*m*xVals/T)
		sum += cosCoeffs[m]*np.cos(2*np.pi*m*xVals/T)
	# Return fSeriesData, a tuple of x and y values
	return xVals, sum

# Use the same dot product method to get fit wellness
def CalcDotProd(gaData, fSeriesData):
	# Steven's function to normalize data
	def normalize(a):
		length=len(a)
		avec=np.array(a, dtype='float')
		norm=np.sqrt(np.sum(avec*avec))
		return avec/norm

	normGA = normalize(gaData[1])
	normfSeries = normalize(fSeriesData[1])

	return np.dot(normGA, normfSeries)

# Plot the results
def Plot(data1, data2, data3, fSin, fCos, fScore, leftXLim=128):
	# Plot the result using matplotlib
	import matplotlib.pyplot as plt
	# Allows for integer ticks in plot's x axis
	from matplotlib.ticker import MaxNLocator

	dat1x, dat1y = data1
	dat2x, dat2y = data2
	dat3x, dat3y = data3
	sinX, sinY = fSin
	cosX, cosY = fCos




	
	# THE CODE BELOW CALCULATES FFT OF REVERSE FFT, COMMENT OUT THIS
	# AND THE ADDITIONAL AX2.SCATTER'S FOR NORMAL PLOTS
	ffSin = np.zeros((len(cosX)))
	ffCos = np.zeros((len(cosX)))

	# Calculate the fourier series coefficients to the NMax'th coefficient
	for i in range(len(cosX)):
		# Store the calculated fourier series in seperate arrays
		# You can change gaData to sinData, to see how much better the 
		# interpolation works on well-sampled and noiseless data
		ffSin[i], ffCos[i] = CalcFourierCoeffs(i, 40, data3)




	# Create a plot with 2 axes
	fig = plt.figure(figsize=(30,8))
	ax1 = fig.add_subplot(1,2,1)

	# Plot progress over generations
	ax1.scatter(dat1x, dat1y, color='green', label='CW', marker='o') 
	ax1.scatter(dat2x, dat2y, color='blue', label='GA Data', \
				marker='o', s= 100)
	ax1.scatter(dat3x, dat3y, color='red', label='Fourier Series '+\
				'Interpolation', marker='o', s= 10)
	ax1.set_xlabel('Time (ns)', fontsize=18)
	ax1.set_xlim(-1, leftXLim)
	ax1.set_ylabel('Amplitude', fontsize=18)
	ax1.set_title('Comparison of CW, Data, and Interpolation with '+ \
				f'Fitness Score {fScore:.4}', fontsize=22)
	ax1.xaxis.set_tick_params(labelsize=20)
	ax1.yaxis.set_tick_params(labelsize=20)
	ax1.legend()
	# Force integer ticks
	ax1.xaxis.set_major_locator(MaxNLocator(integer=True)) 
	

	ax2 = fig.add_subplot(1,2,2)

	# Plot progress over generations
	ax2.scatter(sinX, sinY, color='green', label='sine', marker='x',\
				s=100) 
	ax2.scatter(cosX, cosY, color='blue', label='cosine', marker='o')
	ax2.scatter(sinX, ffSin, color='red', label='FFsine', marker='x',\
				s=100) 
	ax2.scatter(cosX, ffCos, color='yellow', label='FFcosine', marker='o') 
	ax2.set_xlabel('Coefficient', fontsize=18)
	ax2.set_ylabel('Value', fontsize=18)
	ax2.set_title('Fourier Series Coefficients', fontsize=22)
	ax2.xaxis.set_tick_params(labelsize=20)
	ax2.yaxis.set_tick_params(labelsize=20)
	ax2.legend()
	# Force integer ticks
	ax2.xaxis.set_major_locator(MaxNLocator(integer=True)) 
	plt.savefig('results/Ch5Ev000NCoeffs'+str(len(cosX) -1)+ \
				f'FS{fScore:.4}Plot2.png')
	plt.show()

# Start the clock
start_time = time.time()
# For this data, the frequency is:
freq = 1.2
# The period should always be 40 ns
T = 40
# The discreteness of the CW wave and the fourier interpolation
disc = 10000
# Grab the data from channel 5, event 1 and it's associated GA run
phase, amp, gaData = getData()

# Grab a sine wave that matches it
sinData = getCW(T, freq, amp, phase, num=disc)


# This number is the highest fourier series coefficient that will 
# be calculated. It is currently 10 more than what seems to be the
# optimal solution of 48 = T*freq (for some reason) 
fSeriesNMax =  int(round(T*freq))

# Create empty arrays to hold the sine and cosine fourier coeffs
# Note, the zeroth sine coefficient is always 0, and fCos[0]=a_0
fSin = np.zeros((fSeriesNMax+1))
fCos = np.zeros((fSeriesNMax+1))

# Calculate the fourier series coefficients to the NMax'th coefficient
for i in range(fSeriesNMax+1):
	# Store the calculated fourier series in seperate arrays
	# You can change gaData to sinData, to see how much better the 
	# interpolation works on well-sampled and noiseless data
	fSin[i], fCos[i] = CalcFourierCoeffs(i, T, gaData)

# For plotting, create data pairs out of the coefficients
fSeriesX = np.arange(fSeriesNMax+1)
fSeriesSinData = fSeriesX, fSin
fSeriesCosData = fSeriesX, fCos

# We now plot the result of the fourier series. To do this, we use the 
# coefficients and reconstruct the data with num discreteness
fSeriesData = FourierSeries(fSin, fCos, T, num=disc)

# Print how long the actual code took
print(f"Program took {time.time() - start_time:.3} seconds" )

# Calculate how well we did using the same dot product method used for
# the fitness score. We need the same uneven sampling of the fourier
# series, so we recalculate using gaTimes
gaTimes, _ = gaData
fSeriesUnevenData = FourierSeries(fSin, fCos, T, xVals=gaTimes)
fScore = CalcDotProd(gaData, fSeriesUnevenData)
print(fScore)

# Plot the results
Plot(sinData, gaData, fSeriesData, fSeriesSinData, fSeriesCosData, \
	fScore, leftXLim= 10)