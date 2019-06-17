

import gspread
from oauth2client.service_account import ServiceAccountCredentials


'''
def AddToGoogleSpreadsheet(genMax, popMax, numNodes, fitType, 	\
    					fitBreakdown, valRange, meanVal,	\
    					channel,							\
    					Alg2competitorFrac, 				\
    					nodesCrossed, Alg3competitorFrac, 	\
    					Alg4competitorFrac, mutSizeFrac,	\
    					epsPercent, useAlg5, bestScore):
'''
def toSpreadsheet():


	# use creds to create a client to interact with the Google Drive API
	scope = ['https://spreadsheets.google.com/feeds']
	#scope = ['https://www.googleapis.com/auth/spreadsheets']
	jsonName = 'client_secret.json'
	creds = ServiceAccountCredentials.from_json_keyfile_name(jsonName, scope)
	client = gspread.authorize(creds)
	'''
	# Find a workbook by name and open the first sheet
	# Make sure you use the right name here.
	sheet = client.open("SignalCalib_AutoTrials").sheet1
	
	# Extract and print all of the values
	list_of_hashes = sheet.get_all_records()
	print(list_of_hashes)
	'''
	'''
	useA5 = 0
	if useAlg5:
		useA5 = 1

	header = "genMax, popMax, numNodes, fitType, valRange, meanVal"  \
			+", Channel, Alg1Numb, Alg2Numb, Alg3Numb, Alg4Numb, " 	 \
			+"Alg2CompFrac, nodesCrossed, Alg3CompFrac, Alg4CompFrac"\
			+", mutSizeFrac, useAlg5, epsPercent, bestScore\n"

	data = np.array([genMax, popMax, numNodes, fitType, valRange, 	\
			meanVal, channel, fitBreakdown[0], fitBreakdown[1], 	\
			fitBreakdown[2], fitBreakdown[3], Alg2competitorFrac,	\
			nodesCrossed, Alg3competitorFrac, Alg4competitorFrac,	\
			mutSizeFrac, useA5, epsPercent, bestScore]).astype(float)
	'''


toSpreadsheet()