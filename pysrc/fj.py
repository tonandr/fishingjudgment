'''
    Title: Fishing judgment for ships with voyage information.
    
    @author: Inwoo Chung (gutomitai@gmail.com)
    @since: Sep. 8, 2016
    
    Revision:
        -Sep. 8, 2016
            LatLong, ShipVoyageInfo and ShipVoyageInfoWithDependentFeatures are developed. 
            FishingJudgement is added and being developed. 
'''

import csv
import datetime

import ais

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
        
    # Fishing status.
    isFishing = None # -1: Unknown, 0: Not fishing, 1: Fishing.

class ShipVoyageInfoWithDependentFeatures(ShipVoyageInfo):
    '''
        Ship voyage information with dependent features.
    '''
    
    # Dependencies for previous steering.
    sogDiff = None # Speed difference for a preceding one.
    cogDiff = None # COG difference for a preceding one.
    trueHeadingDiff = None # True heading difference for a preceding one.
      
class FishingJudgment:
    '''
        Fishing judgment module.
    '''
    
    def __init__(self):
        '''
            Constructor.
        '''
    
    def train(self, trainingCSVFilePath):
        '''
            Train.
        '''
        
        # Parse a training csv file.
        shipVoyageInfos = parseTrainingCSVFile(trainingCSVFilePath)
    
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
            count = 1
            
            for row in csvReader:
                aisMessage = row[11]
                
                # Check exception.
                # Conduct checksum.
                if (checkSum(aisMessage) != True):
                    if (DebugFlag):
                        print str(count) + ", Checksum failed."
                    continue
                                                
                # Extract ship voyage info. affecting fishing.
                shipVoyageInfo = ShipVoyageInfo()
                
                # Get a time stamp value and extract year, days for 1 year and seconds for 1 day values.
                shipVoyageInfo.utcTimeStamp = int(row[0])
                
                t = datetime.datetime().fromtimestamp(shipVoyageInfo.utcTimeStamp) # Is it valid?
                
                shipVoyageInfo.year = t.year
                
                # Calculate days for 1 year.
                thisYearFirstDay = datetime.datetime(year=y.year, month=1, day=1)
                diffDateTime = t - thisYearFirstDay
                
                shipVoyageInfo.days = diffDateTime.days
                
                # Calculate seconds for 1 day.
                thisDay = datetime.datatime(year=y.year, month=t.month, day=t.day)
                diffDateTime = t - thisDay
                
                shipVoyageInfo.seconds = diffDateTime.seconds
                
                # Decode a AIS message.
                '''
                    The NMEA message type and sentence relevant information are
                    assumed to be identical, so it isn't necessary to be decoded. 
                '''
                                
                aisMessageSplited = aisMessage.split(',')
                
                # Get a channel type.
                if (aisMessageSplited[4] == 'A'):
                    shipVoyageInfo.channelType = 0
                elif (aisMessageSplited[4] == 'B'):
                    shipVoyageInfo.channelType = 1
                else:
                    shipVoyageInfo.channelType = -1
                    
                # Decode a AIS sentence.
                # Get a padding value.
                padCheckString = aisMessageSplited[6]
                padCheckStringSplited = padCheckString.split('*')
                padding = int(padCheckStringSplited[0])
                
                r = ais.decode(aisMessage[5], padding)

                # Extract features from the decoded AIS sentence information.
                shipVoyageInfo.mmsi = r['mmsi']
                
                shipVoyageInfo.pos.lat = r['x']
                shipVoyageInfo.pos.long = r['y']
                shipVoyageInfo.pos.accuracy = r['position_accuracy']
                
                #shipVoyageInfo.naviStatus = r['nav_status'] # Check it!
                shipVoyageInfo.sog = r['sog'] # Check it!
                shipVoyageInfo.cog = r['cog'] # Check it!
                #shipVoyageInfo.trueHeading = r['true_heading'] # Check it.
        
                shipVoyageInfo.isRAIM = r['raim']
                
                # Get fishing status.
                if (row[12] == "Fishing"):
                    shipVoyageInfo.isFishing = 1
                elif (row[12] == "Not Fishing"): # Check.
                    shipVoyageInfo.isFishing = 0
                else:
                    shipVoyageInfo.isFishing = -1
                 
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
            count = 1
            
            for row in csvReader:
                aisMessage = row[1]
                
                # Check exception.
                # Conduct checksum.
                if (checkSum(aisMessage) != True):
                    if (DebugFlag):
                        print str(count) + ", Checksum failed."
                    continue
                                                
                # Extract ship voyage info. affecting fishing.
                shipVoyageInfo = ShipVoyageInfo()
                
                # Get a time stamp value and extract year, days for 1 year and seconds for 1 day values.
                shipVoyageInfo.utcTimeStamp = int(row[0])
                
                t = datetime.datetime().fromtimestamp(shipVoyageInfo.utcTimeStamp) # Is it valid?
                
                shipVoyageInfo.year = t.year
                
                # Calculate days for 1 year.
                thisYearFirstDay = datetime.datetime(year=y.year, month=1, day=1)
                diffDateTime = t - thisYearFirstDay
                
                shipVoyageInfo.days = diffDateTime.days
                
                # Calculate seconds for 1 day.
                thisDay = datetime.datatime(year=y.year, month=t.month, day=t.day)
                diffDateTime = t - thisDay
                
                shipVoyageInfo.seconds = diffDateTime.seconds
                
                # Decode a AIS message.
                '''
                    The NMEA message type and sentence relevant information are
                    assumed to be identical, so it isn't necessary to be decoded. 
                '''
                                
                aisMessageSplited = aisMessage.split(',')
                
                # Get a channel type.
                if (aisMessageSplited[4] == 'A'):
                    shipVoyageInfo.channelType = 0
                elif (aisMessageSplited[4] == 'B'):
                    shipVoyageInfo.channelType = 1
                else:
                    shipVoyageInfo.channelType = -1
                    
                # Decode a AIS sentence.
                # Get a padding value.
                padCheckString = aisMessageSplited[6]
                padCheckStringSplited = padCheckString.split('*')
                padding = int(padCheckStringSplited[0])
                
                r = ais.decode(aisMessage[5], padding)

                # Extract features from the decoded AIS sentence information.
                shipVoyageInfo.mmsi = r['mmsi']
                
                shipVoyageInfo.pos.lat = r['x']
                shipVoyageInfo.pos.long = r['y']
                shipVoyageInfo.pos.accuracy = r['position_accuracy']
                
                #shipVoyageInfo.naviStatus = r['nav_status'] # Check it!
                shipVoyageInfo.sog = r['sog'] # Check it!
                shipVoyageInfo.cog = r['cog'] # Check it!
                #shipVoyageInfo.trueHeading = r['true_heading'] # Check it.
        
                shipVoyageInfo.isRAIM = r['raim']
                                 
                shipVoyageInfos.append(shipVoyageInfo)
                count = count + 1
        
        return shipVoyageInfos                
            
            
            
        