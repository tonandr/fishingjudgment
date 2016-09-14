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
    
    NUM_FACTORS = 11
    
    def __init__(self, gateway, sc):
        '''
            Constructor.
        '''
        
        # Create the Py4j gateway for Spark Neural Network model connection.
        self.gateway = gateway
    
        # Get spark context.
        self.sc = sc
    
    def train(self, trainingCSVFilePath, numLayers, _numActs):
        '''
            Train.
        '''
        
        # Parse a training csv file.
        shipVoyageInfos = self.parseTrainingCSVFile(trainingCSVFilePath)
        
        # Create a training model using Neural Network via Spark.
        numActs = self.gateway.new_array(self.gateway.jvm.int, numLayers)
        
        for i in range(numLayers):
            numActs[i] = _numActs[i]
        
        self.fjnn = self.gateway.entry_point.getSparkNeuralNetwork(self.sc, numLayers, numActs)
        
        # Pre-processing training data.
        pShipVoyageInfos = self.preprocessTrainingData(shipVoyageInfos)
        
        # Make matrix data for training data.
        X = self.gateway.entry_point.getMatrix(self.NUM_FACTORS, len(pShipVoyageInfos), 0.0)
        Y = self.gateway.entry_point.getMatrix(1, len(pShipVoyageInfos), 0.0) # Check row length.
        
        for i in range(len(pShipVoyageInfos)):
            if debugFlag:
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
        
        # Train.
        self.fjnn.train(X, Y)
        
    def test(self, testingCSVFilePath):
        '''
            Test.
        '''
        
        # Parse a testing csv file.
        shipVoyageInfos = self.parseTestingCSVFile(testingCSVFilePath)
                
        # Pre-processing training data.
        pShipVoyageInfos = self.preprocessTrainingData(shipVoyageInfos)
        
        # Make matrix data for training data.
        X = self.gateway.entry_point.getMatrix(self.NUM_FACTORS, len(pShipVoyageInfos), 0.0)
        
        for i in range(len(pShipVoyageInfos)):
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
        
        # Predict fishing confidence.
        Yhat = self.fjnn.predictProb(X)
        
        # Return predicted fishing confidence values according to input samples' order.
        # Assign fishing confidence values.
        for i in range(len(pShipVoyageInfos)):
            pShipVoyageInfos[i].fishingProb = Yhat.getVal(1, i)
        
        # Reorder results.
        pShipVoyageInfos.sort(cmp=self.__compF2__) # Is it a valid function pass format?
        
        # Get the list for fishing probability.
        result = [v.fishingProb for v in pShipVoyageInfos]
        
        return result
    
    def testAndSave(self, testingCSVFilePath):
        '''
            Test and save result into a csv file.
        '''
        
        # Test.
        results = self.test(testingCSVFilePath)
        
        # Save.
        with open('fishing_confidence_result.csv', 'wb') as resultFile:
            csvWriter = csv.writer(resultFile, delimiter=',')
            
            for v in results:
                csvWriter.writerow([v])
    
    def evaluatePerf(self, trainingCSVFilePath, sampleRatio):
        '''
            Evaluate the performance of the fishing judgment model.
        '''
        
    def evaluateLearningCurve(self, trainingCSVFilePath):
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
                f[i].posVariation = self.calEarthDistance(f[i - 1].pos, f[i].pos)
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
        pLats = np.asfarray([v.pos.lat for v in f if ((v.pos.lat >= 0.0) & (v.pos.lat != 91.0))])
        pLongs = np.asfarray([v.pos.lat for v in f if ((v.pos.long >= 0.0) & (v.pos.long != 181.0))])
        nLats = np.asfarray([v.pos.lat for v in f if (v.pos.lat < 0.0)])
        nLongs = np.asfarray([v.pos.lat for v in f if (v.pos.long < 0.0)])    
        
        self.aPLat = pLats.mean()
        self.aPLong = pLongs.mean()
        self.aNLat = nLats.mean()
        self.aNLong = nLongs.mean()
        
        # SOG.
        sogs = np.asfarray([v.sog for v in f if (v.sog < 102.3)])
        self.aSOG = sogs.mean()
        
        # COG.
        cogs = np.asfarray([v.cog for v in f if (v.cog < 360.0)])
        self.aCOG = cogs.mean()
        
        for i in range(len(f)):
                        
            # Position.
            # Fill missing values.
            if (f[i].pos.lat == 91.0):
                f[i].pos.lat = self.aPLat
            
            if (f[i].pos.long == 181.0):
                f[i].pos.long = self.aPLong
            
            # SOG.
            if (f[i].sog == 102.3):
                f[i].sog = self.aSOG
            
            # COG.
            if (f[i].cog >= 360.0):
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
        
    def parseTrainingCSVFile(self, trainingCSVFilePath):
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
                
                if (debugFlag):
                    print str(count) + ": "  + aisMessage
                
                # Check exception.
                # Conduct checksum.
                if (self.__checksum__(aisMessage) != True):
                    if (DebugFlag):
                        print str(count) + ", Checksum failed."
                    continue
                                                
                # Extract ship voyage info. affecting fishing.
                shipVoyageInfo = ShipVoyageInfo()
                
                # Get a time stamp value and extract year, days for 1 year and seconds for 1 day values.
                shipVoyageInfo.utcTimeStamp = float(row[0])
                
                t = datetime.datetime.fromtimestamp(shipVoyageInfo.utcTimeStamp)
                
                shipVoyageInfo.year = float(t.year)
                
                # Calculate days for 1 year.
                thisYearFirstDay = datetime.datetime(year=t.year, month=1, day=1)
                diffDateTime = t - thisYearFirstDay
                
                shipVoyageInfo.days = float(diffDateTime.days)
                
                # Calculate seconds for 1 day.
                thisDay = datetime.datetime(year=t.year, month=t.month, day=t.day)
                diffDateTime = t - thisDay
                
                shipVoyageInfo.seconds = float(diffDateTime.seconds)
                
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
                
                shipVoyageInfo.pos.lat = float(r['x'])
                shipVoyageInfo.pos.long = float(r['y'])
                shipVoyageInfo.pos.accuracy = float(r['position_accuracy'])
                
                #shipVoyageInfo.naviStatus = r['nav_status'] # Check it!
                shipVoyageInfo.sog = float(r['sog']) # Check it!
                shipVoyageInfo.cog = float(r['cog']) # Check it!
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
        
        return shipVoyageInfos        
                
    def parseTestingCSVFile(self, testingCSVFilePath):
        '''
            Parse a testing csv file.
        '''
        
        if (debugFlag):
            print "Parse a testing csv file..."
        
        # Create the ship voyage info. list.
        shipVoyageInfos = list()
        
        # Read a csv file.
        with open(trainingCSVFilePath, 'rb') as csvFile:
            csvReader = csv.reader(csvFile, delimiter=',', quotechar='"' )
            csvReader.next()
            
            count = 1
            
            for row in csvReader:
                aisMessage = row[1]
                
                if (debugFlag):
                    print str(count) + ": "  + aisMessage
                
                # Check exception.
                # Conduct checksum.
                if (self.__checksum__(aisMessage) != True):
                    if (DebugFlag):
                        print str(count) + ", Checksum failed."
                                                
                # Extract ship voyage info. affecting fishing.
                shipVoyageInfo = ShipVoyageInfo()
                
                # Get a time stamp value and extract year, days for 1 year and seconds for 1 day values.
                shipVoyageInfo.utcTimeStamp = float(row[0])
                
                t = datetime.datetime.fromtimestamp(shipVoyageInfo.utcTimeStamp)
                
                shipVoyageInfo.year = float(t.year)
                
                # Calculate days for 1 year.
                thisYearFirstDay = datetime.datetime(year=t.year, month=1, day=1)
                diffDateTime = t - thisYearFirstDay
                
                shipVoyageInfo.days = float(diffDateTime.days)
                
                # Calculate seconds for 1 day.
                thisDay = datetime.datetime(year=t.year, month=t.month, day=t.day)
                diffDateTime = t - thisDay
                
                shipVoyageInfo.seconds = float(diffDateTime.seconds)
                
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
                
                shipVoyageInfo.pos.lat = float(r['x'])
                shipVoyageInfo.pos.long = float(r['y'])
                shipVoyageInfo.pos.accuracy = float(r['position_accuracy'])
                
                #shipVoyageInfo.naviStatus = r['nav_status'] # Check it!
                shipVoyageInfo.sog = float(r['sog']) # Check it!
                shipVoyageInfo.cog = float(r['cog']) # Check it!
                #shipVoyageInfo.trueHeading = r['true_heading'] # Check it.
        
                shipVoyageInfo.isRAIM = r['raim']
                
                shipVoyageInfo.index = count
                                 
                shipVoyageInfos.append(shipVoyageInfo)
                count = count + 1
        
        return shipVoyageInfos                
            
            
            
        