'''
    Title: Fishing judgment for ships with voyage information.
    
    @author: Inwoo Chung (gutomitai@gmail.com)
    @since: Sep. 8, 2016
    
    Revision:
        -Sep. 8, 2016
            LatLong, ShipVoyageInfo and ShipVoyageInfoWithDependentFeatures are developed. 
            FishingJudgement is added and being developed.
        -Sep. 14, 2016
            Development from test to train has completed.
        -Sep. 16, 2016
            Optimization methods are applied.
        -Sep. 17, 2016
            Gradient checking is added.
        -Oct. 4, 2016
            AUC, Gini coefficient and score calculation portions are added for model performance evaluation.
'''

import csv
import datetime
import math

import numpy as np
import ais
from py4j.java_gateway import JavaGateway

# Debug flag.
debugFlag = True

class LatLong:
    '''
        Latitude and longitude with an accuracy flag.
    '''
    
    lat = None # Latitude.
    long = None # Longitude.
    accuracy = None # Accuracy: 0, 1.

class ShipVoyageInfo:
    '''
        Ship voyage Information affecting fishing.
    '''
    
    # Index.
    index = None
    
    # Communication status.
    channelType = None # Channel type: A:0, B:1.
    isRAIM = None # Whether RAIM is used or not: false, true.
    
    # Ship identity.
    mmsi = None # Maritime Mobile Ship Identity. 
    
    # Time.
    utcTimeStamp = None # UTC time stamp.
    year = None # Year.
    days = None # Days for 1 year.
    seconds = None # Seconds for 1 day.
    
    # Position.
    '''
    The position accuracy flag indicates the accuracy of the fix. 
    A value of 1 indicates a DGPS-quality fix with an accuracy of < 10ms. 0, the default, 
    indicates an unaugmented GNSS fix with accuracy > 10m.

    Longitude is given in in 1/10000 min; divide by 600000.0 to obtain degrees. 
    Values up to plus or minus 180 degrees, East = positive, West \= negative. 
    A value of 181 degrees (0x6791AC0 hex) indicates that longitude is not available and is the default.

    Latitude is given in in 1/10000 min; divide by 600000.0 to obtain degrees. 
    Values up to plus or minus 90 degrees, North = positive, South = negative. 
    A value of 91 degrees (0x3412140 hex) indicates latitude is not available and is the default.
    
    This is from http://catb.org/gpsd/AIVDM.html#_types_1_2_and_3_position_report_class_a.
    '''
    pos = LatLong() # Position with accuracy.
    
    # Ship steering features.
    naviStatus = None # Navigation status: 0 ~ 15.
    
    # Speed over ground.
    '''
    Speed over ground in 0.1-knot resolution from 0 to 102 knots. 
    Value 1023 indicates speed is not available, value 1022 indicates 102.2 knots or higher.
    
    This is from http://catb.org/gpsd/AIVDM.html#_types_1_2_and_3_position_report_class_a.
    '''
    sog = None 
    
    # Course over ground.
    '''
    Relative to true north, to 0.1 degree precision.
    Course over ground will be 3600 (0xE10) if that data is not available.
    
    This is from http://catb.org/gpsd/AIVDM.html#_types_1_2_and_3_position_report_class_a.
    '''
    cog = None # Course over ground.
    
    # True heading.
    '''
    0 to 359 degrees, 511 = not available.
    
    This is from http://catb.org/gpsd/AIVDM.html#_types_1_2_and_3_position_report_class_a.
    '''
    trueHeading = None # True heading.
    
    # Dependencies for previous steering.
    posVariation = None # Distance between a current position and a previous position.
    sogDiff = None # Speed difference for a preceding one.
    cogDiff = None # COG difference for a preceding one.
    
    trueHeadingDiff = None # True heading difference for a preceding one.
        
    # Fishing status.
    isFishing = None # -1: Unknown, 0: Not fishing, 1: Fishing.
    
    # Fishing confidence.
    fishingProb = None
          
class FishingJudgment:
    '''
        Fishing judgment module.
    '''
        
    X_FILE_NAME = "X.ser"
    Y_FILE_NAME = "Y.ser"
    
    TRAINING_X_FILE_NAME = "TR_X.ser"
    TESTING_X_FILE_NAME = "TE_X.ser"
    
    NUM_FACTORS = 11
    
    def __init__(self, gateway, sc):
        '''
            Constructor.
        '''
        
        # Create the Py4j gateway for Spark Neural Network model connection.
        self.gateway = gateway
    
        # Get spark context.
        self.sc = sc
    
    def train(self, trainingCSVFilePath, numLayers, _numActs, opt, l, alpha, loadFlag = False, isGradientChecking = False, JEstimationFlag = False, sampleRatio = 1.0):
        '''
            Train.
        '''
        
        # Check the load flag.
        if (loadFlag == True):
            
            # Create a training model using Neural Network via Spark.
            numActs = self.gateway.new_array(self.gateway.jvm.int, numLayers)
        
            for i in range(numLayers):
                numActs[i] = _numActs[i]
        
            self.fjnn = self.gateway.entry_point.getSparkNeuralNetwork(numLayers, numActs, opt)
            
            # Get X, Y.
            RX = self.gateway.jvm.maum.dm.Matrix.loadMatrix(self.X_FILE_NAME)
            RY = self.gateway.jvm.maum.dm.Matrix.loadMatrix(self.Y_FILE_NAME)
            
            # Calculate the number of samples for training according to sample ratio.
            # RX is assumed to have full samples.
            numSamples = int(RX.colLength() * sampleRatio)
            
            # Extract X, Y from RX, RY.
            xRange = self.gateway.new_array(self.gateway.jvm.int, 4)
            yRange = self.gateway.new_array(self.gateway.jvm.int, 4)
            
            xRange[0] = 1; xRange[1] = RX.rowLength(); xRange[2] = 1; xRange[3] = numSamples
            yRange[0] = 1; yRange[1] = RY.rowLength(); yRange[2] = 1; yRange[3] = numSamples
            
            X = RX.getSubMatrix(xRange)
            Y = RY.getSubMatrix(yRange)
            
            # Train.
            return self.fjnn.train(self.sc, X, Y, l, alpha, isGradientChecking, JEstimationFlag)
            
        # Parse a training csv file.
        shipVoyageInfos = self.parseTrainingCSVFile(trainingCSVFilePath, sampleRatio)
        
        # Create a training model using Neural Network via Spark.
        numActs = self.gateway.new_array(self.gateway.jvm.int, numLayers)
        
        for i in range(numLayers):
            numActs[i] = _numActs[i]
        
        self.fjnn = self.gateway.entry_point.getSparkNeuralNetwork(numLayers, numActs, opt)
        
        # Pre-processing training data.
        pShipVoyageInfos = self.preprocessTrainingData(shipVoyageInfos)
        
        # Get the number of samples.
        numSamples = int(len(pShipVoyageInfos))
        
        # Make matrix data for training data.
        X = self.gateway.entry_point.getMatrix(self.NUM_FACTORS, numSamples, 0.0)
        Y = self.gateway.entry_point.getMatrix(1, numSamples, 0.0) # Check row length.
        
        for i in range(numSamples):
            if debugFlag&False:
                print str(i)
                
            v = pShipVoyageInfos[i]
            
            X.setVal(1, i + 1, v.year)
            X.setVal(2, i + 1, v.days)
            X.setVal(3, i + 1, v.seconds)
            X.setVal(4, i + 1, v.pos.lat)
            X.setVal(5, i + 1, v.pos.long)
            X.setVal(6, i + 1, v.pos.accuracy)
            X.setVal(7, i + 1, v.sog)
            X.setVal(8, i + 1, v.cog)
            X.setVal(9, i + 1, v.posVariation)
            X.setVal(10, i + 1, v.sogDiff)
            X.setVal(11, i + 1, v.cogDiff)
            
            Y.setVal(1, i + 1, v.isFishing)
        
        # Save X, Y.
        X.saveMatrix(self.X_FILE_NAME)
        Y.saveMatrix(self.Y_FILE_NAME)
        
        # Train.
        return self.fjnn.train(self.sc, X, Y, l, alpha, isGradientChecking, JEstimationFlag)
    
    def test(self, CSVFilePath, sampleRatio, loadFlag=False, isTrainingData=False):
        '''
            Test.
        '''
        
        shipVoyageInfos = None; 
        
        # Check the load flag.
        if (loadFlag == True):
            
            # Parse a testing csv file.
            if (isTrainingData):
                shipVoyageInfos = self.parseTrainingCSVFile(CSVFilePath, sampleRatio)
            else:
                self.parseTestingCSVFile(CSVFilePath, sampleRatio)
                
            # Pre-processing training data.
            pShipVoyageInfos = self.preprocessTrainingData(shipVoyageInfos)
                        
            # Get X.
            RX = None
            
            if (isTrainingData):
                RX = self.gateway.jvm.maum.dm.Matrix.loadMatrix(self.TRAINING_X_FILE_NAME)
            else:
                RX = self.gateway.jvm.maum.dm.Matrix.loadMatrix(self.TESTING_X_FILE_NAME)
    
            # RX is assumed to have full samples.
            numSamples = int(RX.colLength() * sampleRatio)
                        
            # Extract X from RX.
            xRange = self.gateway.new_array(self.gateway.jvm.int, 4)
            xRange[0] = 1; xRange[1] = RX.rowLength(); xRange[2] = 1; xRange[3] = numSamples

            X = RX.getSubMatrix(xRange)
            
            # Predict fishing confidence.
            Yhat = self.fjnn.predictProb(X)
        
            # Return predicted fishing confidence values according to input samples' order.
            # Assign fishing confidence values.
            for i in range(numSamples):
                pShipVoyageInfos[i].fishingProb = Yhat.getVal(1, i + 1)
        
            # Reorder results.
            pShipVoyageInfos.sort(cmp=self.__compF2__) # Is it a valid function pass format?
        
            # Get the list for fishing probability.
            result = [v.fishingProb for v in pShipVoyageInfos]
        
            return (result, pShipVoyageInfos)
        
        # Parse a testing csv file.
        if (isTrainingData):
            shipVoyageInfos = self.parseTrainingCSVFile(CSVFilePath, sampleRatio)
        else:
            self.parseTestingCSVFile(CSVFilePath, sampleRatio)
                
        # Pre-processing testing data.
        pShipVoyageInfos = self.preprocessTrainingData(shipVoyageInfos)
        
        # Make matrix data for testing data.
        RX = self.gateway.entry_point.getMatrix(self.NUM_FACTORS, len(pShipVoyageInfos), 0.0)
        
        for i in range(len(pShipVoyageInfos)):
            v = pShipVoyageInfos[i]
            
            RX.setVal(1, i + 1, v.year)
            RX.setVal(2, i + 1, v.days)
            RX.setVal(3, i + 1, v.seconds)
            RX.setVal(4, i + 1, v.pos.lat)
            RX.setVal(5, i + 1, v.pos.long)
            RX.setVal(6, i + 1, v.pos.accuracy)
            RX.setVal(7, i + 1, v.sog)
            RX.setVal(8, i + 1, v.cog)
            RX.setVal(9, i + 1, v.posVariation)
            RX.setVal(10, i + 1, v.sogDiff)
            RX.setVal(11, i + 1, v.cogDiff)
        
        # Save X.
        if (isTrainingData):
            RX.saveMatrix(self.TRAINING_X_FILE_NAME)
        else:
            RX.saveMatrix(self.TESTING_X_FILE_NAME)
        
        # Predict fishing confidence.
        # Calculate the number of samples and extract X according to the number of samples.
        numSamples = int(len(pShipVoyageInfos) * sampleRatio)
        
        # Extract X from RX.
        xRange = self.gateway.new_array(self.gateway.jvm.int, 4)
        xRange[0] = 1; xRange[1] = RX.rowLength(); xRange[2] = 1; xRange[3] = numSamples

        X = RX.getSubMatrix(xRange)
        
        Yhat = self.fjnn.predictProb(X)
        
        # Return predicted fishing confidence values according to input samples' order.
        # Assign fishing confidence values.
        for i in range(numSamples):
            pShipVoyageInfos[i].fishingProb = Yhat.getVal(1, i + 1)
        
        # Reorder results.
        pShipVoyageInfos.sort(cmp=self.__compF2__) # Is it a valid function pass format?
        
        # Get the list for fishing probability.
        result = [v.fishingProb for v in pShipVoyageInfos]
        
        return (result, pShipVoyageInfos)
    
    def testAndSave(self, CSVFilePath, sampleRatio, loadFlag = False, isTrainingData=False):
        '''
            Test and save result into a csv file.
        '''
        
        # Test.
        results = self.test(CSVFilePath, sampleRatio, loadFlag, isTrainingData)
        result = results[0]
        pShipVoyageInfos = results[1]
        
        # Assign 0.0 for samples with checksum error.        
        for i in range(len(pShipVoyageInfos)):
            if (pShipVoyageInfos[i].utcTimeStamp == -1.0):
                result[i] = 0.0
        
        # Save.
        with open('fishing_confidence_result.csv', 'wb') as resultFile:
            csvWriter = csv.writer(resultFile, delimiter=',')
            
            for v in result:
                csvWriter.writerow([v])
    
    def evaluatePerf(self, trainingCSVFilePath, sampleRatio, loadFlag = False):
        '''
            Evaluate the performance of the fishing judgment model.
        '''
        
        # Test with training samples.
        results = self.test(trainingCSVFilePath, sampleRatio, loadFlag, isTrainingData = True)
        result = results[0]
        pShipVoyageInfos = results[1]
        
        # Assign 0.0 for samples with checksum error.        
        for i in range(len(pShipVoyageInfos)):
            if (pShipVoyageInfos[i].utcTimeStamp == -1.0):
                result[i]= 0
        
        # Get a ground truth list.
        groundTruths = list()
        
        for v in pShipVoyageInfos:
            if v.isFishing == True:
                groundTruths.append(1.0)
            else:
                groundTruths.append(0.0)
        
        # Calculate a Gini coefficient value.
        # Calculate the area under ROC approximately.
        # Sort the result and ground truth lists together into descending order.
        for i in range(len(result)):
            for j in range(i + 1, len(result)):
                if (result[i] < result[j]):
                    tempResult = result[i]
                    result[i] = result[j]
                    result[j] = tempResult
                    
                    tempGT = groundTruths[i]
                    groundTruths[i] = groundTruths[j]
                    groundTruths[j] = tempGT 
        
        # Calculate TPRs and FPRs.
        numTPRs = float(groundTruths.count(1.0))
        numFPRs = float(groundTruths.count(0.0))
        
        TPRs = list()
        FPRs = list()
        
        for v in groundTruths:
            TPRs.append(v / TPRs)
            FPRs.append((v - 1.0) / FPRs)
        
        # Calculate AUC.
        AUC = 0.0
        
        for i in range(len(TPRs)):
            if i == 0:
                AUC = AUC + TPRs[i] * (FPRs[i])
            else:
                AUC = AUC + TPRs[i] * (FPRs[i] - FPRs[i - 1])
                
        # Calculate Gini coefficent.
        gini = 2.0 * AUC - 1.0
        
        # Calculate a score.
        score = 1000000.0 * gini
            
        print score
        return score
        
    def evaluateLearningCurve(self, trainingCSVFilePath, trainingSampleRatio, sampleRatioStep):
        '''
            Evaluate learning curve.
        '''
                
    def preprocessTrainingData(self, shipVoyageInfos): # Check it!
        '''
            Preprocess training data.
        '''
        
        # Filter data with unknown as fishing state.
        fShipVoyageInfos = list()
        
        for v in shipVoyageInfos:
            if v.isFishing != -1:
                fShipVoyageInfos.append(v)
        
        # Sort it with ascending for time.
        fShipVoyageInfos.sort(cmp=self.__compF__)
        
        # Fill missing factors.
        self.fillMissingFactors(fShipVoyageInfos)
        
        # Calculate and fill difference values about SOG and COG.
        self.calFillDiff(fShipVoyageInfos)
        
        return fShipVoyageInfos
    
    def calFillDiff(self, fSortedShipVoyageInfos):
        '''
            Calculate and fill difference values about position variation , SOG and COG.
        '''
        
        f = fSortedShipVoyageInfos
        currentMMSI = -1        
                
        for i in range(len(f)):
            
            # Check mmsi is changed.
            if (f[i].mmsi != currentMMSI):
                currentMMSI = f[i].mmsi
                
                f[i].posVariation = 0.0; # Is it valid?
                f[i].sogDiff = self.aSOG - f[i].sog
                f[i].cogDiff = self.aCOG - f[i].cog
            else:               
                f[i].posVariation = 0.0 #self.calEarthDistance(f[i - 1].pos, f[i].pos)
                f[i].sogDiff = f[i - 1].sog - f[i].sog
                f[i].cogDiff = f[i - 1].cog - f[i].cog
    
    def calEarthDistance(self, p1, p2):
        '''
            Calculate the distance between p1 and p2 on the Earth using Haversine formula.
        '''
        
        R = 6371.01
        d = 2.0 * R * math.asin(math.sqrt(math.pow(math.sin((math.radians(p2.lat) - math.radians(p2.lat)/2.0)), 2.0) \
                                          + math.cos(math.radians(p1.lat)) * math.cos(math.radians(p2.lat))
                                          * math.pow(math.sin((math.radians(p2.long) - math.radians(p2.long)/2.0)), 2.0)
                                          ))
        
        return d
    
    def fillMissingFactors(self, fSortedShipVoyageInfos):
        '''
            Fill missing factors.
        '''
        f = fSortedShipVoyageInfos
        
        # Calculate average values for missing factors.
        # position.
        pLats = np.asfarray([v.pos.lat for v in f if ((v.pos.lat >= 0.0) & (v.pos.lat != (91.0 / 90.0)))])
        pLongs = np.asfarray([v.pos.long for v in f if ((v.pos.long >= 0.0) & (v.pos.long != (181.0 / 180.0)))])
        nLats = np.asfarray([v.pos.lat for v in f if (v.pos.lat < 0.0)])
        nLongs = np.asfarray([v.pos.long for v in f if (v.pos.long < 0.0)])    
        
        if pLats.size == 0:
            self.aPlat = 0.0
        else:
            self.aPLat = pLats.mean()
            
        if pLongs.size == 0:
            self.aPLongs = 0.0
        else:
            self.aPLongs = pLongs.mean()
                
        if nLats.size == 0:
            self.nPlat = 0.0
        else:
            self.nPLat = nLats.mean()
            
        if nLongs.size == 0:
            self.nPLongs = 0.0
        else:
            self.nPLongs = nLongs.mean()
        
        # SOG.
        sogs = np.asfarray([v.sog for v in f if (v.sog < (102.3 / 102.2))])
        self.aSOG = sogs.mean()
        
        # COG.
        cogs = np.asfarray([v.cog for v in f if (v.cog < (360.0 / 360.0))])
        self.aCOG = cogs.mean()
        
        for i in range(len(f)):
                        
            # Position.
            # Fill missing values.
            if (f[i].pos.lat == (91.0 / 90.0)):
                f[i].pos.lat = self.aPLat
            
            if (f[i].pos.long == (181.0 / 180.0)):
                f[i].pos.long = self.aPLong
            
            # SOG.
            if (f[i].sog == (102.3 / 102.2)):
                f[i].sog = self.aSOG
            
            # COG.
            if (f[i].cog >= (360.0 / 360.0)):
                f[i].cog = self.aCOG   
                
    def __compF__(self, x, y):
        '''
            Comparing function to sort ship voyage information lists with ascending for time.
        '''
        
        if (x.utcTimeStamp > y.utcTimeStamp):
            return 1
        elif (x.utcTimeStamp == y.utcTimeStamp):
            return 0
        else:
            return -1
    
    def __compF2__(self, x, y):
        '''
            Comparing function to sort ship voyage information lists with ascending for index.
        '''
        
        if (x.index > y.index):
            return 1
        elif (x.index == y.index):
            return 0
        else:
            return -1    
    
    def __checksum__(self, aisMessage):
        '''
            AIS message checksum.
        '''
                        
        # Get a checksum value.
        aisMessageSplited = aisMessage.split(',')
        padCheckString = aisMessageSplited[6]
        padCheckStringSplited = padCheckString.split('*')
        checksumVal = int(float.fromhex(padCheckStringSplited[1]))
        
        # Calculate the checksum value for the AIS message.
        # Extract a valid sentence except for ! and *.
        validAISMessage = ((aisMessage[1:]).split('*'))[0]
        
        # Get the byte array for it.
        bValidAISMessage = bytearray(validAISMessage)
        
        # Calculate checksum.
        checksum = 0
        
        for v in bValidAISMessage:
            checksum = checksum^v
        
        if (checksum != checksumVal):
            return False
        
        return True
        
    def parseTrainingCSVFile(self, trainingCSVFilePath, sampleRatio):
        '''
            Parse a training csv file.
        '''
        
        if (debugFlag):
            print "Parse a training csv file..."
        
        # Create the ship voyage info. list.
        shipVoyageInfos = list()
        
        # Read a csv file.
        with open(trainingCSVFilePath, 'rb') as csvFile:
            csvReader = csv.reader(csvFile, delimiter=',', quotechar='"' )
            csvReader.next()
            
            count = 1
            
            for row in csvReader:   
                aisMessage = row[11]
                
                if (debugFlag&False):
                    print str(count) + ": "  + aisMessage
                
                # Check exception.
                # Conduct checksum.
                isCheckSum = True
                
                if (self.__checksum__(aisMessage) != True):
                    if (debugFlag):
                        print str(count) + ", Checksum failed."
                    isCheckSum = False
                
                if (isCheckSum == False):    
                    shipVoyageInfo = ShipVoyageInfo()
                
                    shipVoyageInfo.utcTimeStamp = -1.0                
                    shipVoyageInfo.year = 0.0               
                    shipVoyageInfo.days = 0.0              
                    shipVoyageInfo.seconds = 0.0
                
                    shipVoyageInfo.channelType = float(-1)
                    
                    shipVoyageInfo.mmsi = 0.0
                
                    shipVoyageInfo.pos.lat = 91.0 / 90.0
                    shipVoyageInfo.pos.long = 181.0 / 180.0
                    shipVoyageInfo.pos.accuracy = 0.0
                
                    #shipVoyageInfo.naviStatus = r['nav_status'] # Check it!
                    shipVoyageInfo.sog = 511.0 / 102.2
                    shipVoyageInfo.cog = 360.0 / 360.0
                    #shipVoyageInfo.trueHeading = r['true_heading'] # Check it.
        
                    shipVoyageInfo.isRAIM = False
                
                    shipVoyageInfo.index = count
                                 
                    shipVoyageInfos.append(shipVoyageInfo)
                    count = count + 1
                    
                    continue
                                                
                # Extract ship voyage info. affecting fishing.
                shipVoyageInfo = ShipVoyageInfo()
                
                # Get a time stamp value and extract year, days for 1 year and seconds for 1 day values.
                shipVoyageInfo.utcTimeStamp = float(row[0])
                
                t = datetime.datetime.fromtimestamp(shipVoyageInfo.utcTimeStamp)
                
                shipVoyageInfo.year = 0.0 #float(t.year)
                
                # Calculate days for 1 year.
                thisYearFirstDay = datetime.datetime(year=t.year, month=1, day=1)
                diffDateTime = t - thisYearFirstDay
                
                shipVoyageInfo.days = float(diffDateTime.days) / 365.0
                
                # Calculate seconds for 1 day.
                thisDay = datetime.datetime(year=t.year, month=t.month, day=t.day)
                diffDateTime = t - thisDay
                
                shipVoyageInfo.seconds = float(diffDateTime.seconds) / (24.0 * 3600.0)
                
                # Decode a AIS message.
                '''
                    The NMEA message type and sentence relevant information are
                    assumed to be identical, so it isn't necessary to be decoded. 
                '''
                                
                aisMessageSplited = aisMessage.split(',')
                
                # Get a channel type.
                if (aisMessageSplited[4] == 'A'):
                    shipVoyageInfo.channelType = float(0)
                elif (aisMessageSplited[4] == 'B'):
                    shipVoyageInfo.channelType = float(1)
                else:
                    shipVoyageInfo.channelType = float(-1)
                    
                # Decode a AIS sentence.
                # Get a padding value.
                padCheckString = aisMessageSplited[6]
                padCheckStringSplited = padCheckString.split('*')
                padding = int(padCheckStringSplited[0])
                
                r = ais.decode(aisMessageSplited[5], padding)

                # Extract features from the decoded AIS sentence information.
                shipVoyageInfo.mmsi = float(r['mmsi'])
                
                shipVoyageInfo.pos.lat = float(r['x']) / 90.0
                shipVoyageInfo.pos.long = float(r['y']) /180.0
                shipVoyageInfo.pos.accuracy = float(r['position_accuracy'])
                
                #shipVoyageInfo.naviStatus = r['nav_status'] # Check it!
                shipVoyageInfo.sog = float(r['sog']) / 102.2 # Check it! 
                shipVoyageInfo.cog = float(r['cog']) / 360.0 # Check it!
                #shipVoyageInfo.trueHeading = r['true_heading'] # Check it.
        
                shipVoyageInfo.isRAIM = r['raim']
                
                # Get fishing status.
                if (row[12] == "Fishing"):
                    shipVoyageInfo.isFishing = float(1)
                elif (row[12] == "Not Fishing"): # Check.
                    shipVoyageInfo.isFishing = float(0)
                else:
                    shipVoyageInfo.isFishing = float(-1)
                
                shipVoyageInfo.index = count
                 
                shipVoyageInfos.append(shipVoyageInfo)
                count = count + 1
        
        # Get samples according to sample ratio.
        numSamples = int(len(shipVoyageInfos) * sampleRatio) 
        
        return [shipVoyageInfos[i] for i in range(numSamples)]        
                
    def parseTestingCSVFile(self, testingCSVFilePath):
        '''
            Parse a testing csv file.
        '''
        
        if (debugFlag):
            print "Parse a testing csv file..."
        
        # Create the ship voyage info. list.
        shipVoyageInfos = list()
        
        # Read a csv file.
        with open(testingCSVFilePath, 'rb') as csvFile:
            csvReader = csv.reader(csvFile, delimiter=',', quotechar='"' )
            csvReader.next()
            
            count = 1
            
            for row in csvReader:
                aisMessage = row[1]
                
                if (debugFlag&False):
                    print str(count) + ": "  + aisMessage
                
                # Check exception.
                # Conduct checksum.
                isCheckSum = True
                
                if (self.__checksum__(aisMessage) != True):
                    if (debugFlag):
                        print str(count) + ", Checksum failed."
                        isCheckSum = False
                
                if (isCheckSum == False):
                    
                    shipVoyageInfo = ShipVoyageInfo()
                
                    shipVoyageInfo.utcTimeStamp = -1.0                
                    shipVoyageInfo.year = 0.0               
                    shipVoyageInfo.days = 0.0              
                    shipVoyageInfo.seconds = 0.0
                
                    shipVoyageInfo.channelType = float(-1)
                    
                    shipVoyageInfo.mmsi = 0.0
                
                    shipVoyageInfo.pos.lat = 91.0 / 90.0
                    shipVoyageInfo.pos.long = 181.0 / 180.0
                    shipVoyageInfo.pos.accuracy = 0.0
                
                    #shipVoyageInfo.naviStatus = r['nav_status'] # Check it!
                    shipVoyageInfo.sog = 511.0 / 102.2
                    shipVoyageInfo.cog = 360.0 / 360.0
                    #shipVoyageInfo.trueHeading = r['true_heading'] # Check it.
        
                    shipVoyageInfo.isRAIM = False
                
                    shipVoyageInfo.index = count
                                 
                    shipVoyageInfos.append(shipVoyageInfo)
                    count = count + 1
                    
                    continue
                                                
                # Extract ship voyage info. affecting fishing.
                shipVoyageInfo = ShipVoyageInfo()
                
                # Get a time stamp value and extract year, days for 1 year and seconds for 1 day values.
                shipVoyageInfo.utcTimeStamp = float(row[0])
                
                t = datetime.datetime.fromtimestamp(shipVoyageInfo.utcTimeStamp)
                
                shipVoyageInfo.year = 0.0 #float(t.year)
                
                # Calculate days for 1 year.
                thisYearFirstDay = datetime.datetime(year=t.year, month=1, day=1)
                diffDateTime = t - thisYearFirstDay
                
                shipVoyageInfo.days = float(diffDateTime.days) / 360.0
                
                # Calculate seconds for 1 day.
                thisDay = datetime.datetime(year=t.year, month=t.month, day=t.day)
                diffDateTime = t - thisDay
                
                shipVoyageInfo.seconds = float(diffDateTime.seconds) / (24.0 * 3600.0)
                
                # Decode a AIS message.
                '''
                    The NMEA message type and sentence relevant information are
                    assumed to be identical, so it isn't necessary to be decoded. 
                '''
                                
                aisMessageSplited = aisMessage.split(',')
                
                # Get a channel type.
                if (aisMessageSplited[4] == 'A'):
                    shipVoyageInfo.channelType = float(0)
                elif (aisMessageSplited[4] == 'B'):
                    shipVoyageInfo.channelType = float(1)
                else:
                    shipVoyageInfo.channelType = float(-1)
                    
                # Decode a AIS sentence.
                # Get a padding value.
                padCheckString = aisMessageSplited[6]
                padCheckStringSplited = padCheckString.split('*')
                padding = int(padCheckStringSplited[0])
                
                r = ais.decode(aisMessageSplited[5], padding)

                # Extract features from the decoded AIS sentence information.
                shipVoyageInfo.mmsi = float(r['mmsi'])
                
                shipVoyageInfo.pos.lat = float(r['x']) / 90.0
                shipVoyageInfo.pos.long = float(r['y']) / 180.0
                shipVoyageInfo.pos.accuracy = float(r['position_accuracy'])
                
                #shipVoyageInfo.naviStatus = r['nav_status'] # Check it!
                shipVoyageInfo.sog = float(r['sog']) / 102.2# Check it!
                shipVoyageInfo.cog = float(r['cog']) / 360.0 # Check it!
                #shipVoyageInfo.trueHeading = r['true_heading'] # Check it.
        
                shipVoyageInfo.isRAIM = r['raim']
                
                shipVoyageInfo.index = count
                                 
                shipVoyageInfos.append(shipVoyageInfo)
                count = count + 1
        
        # Get samples according to sample ratio.
        numSamples = int(len(shipVoyageInfos) * sampleRatio) 
        
        return [shipVoyageInfos[i] for i in range(numSamples)]                  
            
            
            
        