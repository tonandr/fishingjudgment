'''
    Title: Analysis about fishing judgment for ships with voyage information.
    
    @author: Inwoo Chung (gutomitai@gmail.com)
    @since: Sep. 9, 2016
    
    Revision:
        -Sep. 9, 2016
'''

def convShipVInfosToDictList(shipVInfos):
    '''
        Convert a ship voyage information list into a dictionary list.
    '''
    
    # Convert.
    shipVInfoDicts = list()
    
    for v in shipVInfos:
        shipVInfoDict = {'channelType': v.channelType, \
                         'isRAIM': v.isRAIM, \
                         'mmsi': v.mmsi, \
                         'utcTimeStamp': v.utcTimeStamp, \
                         'year': v.year, \
                         'days': v.days, \
                         'seconds': v.seconds, \
                         'lat': v.pos.lat, \
                         'long': v.pos.long, \
                         'accuracy': v.pos.accuracy, \
                         #'naviStatus': v.naviStatus, \
                         'sog': v.sog, \
                         'cog': v.cog, \
                         #'trueHeading': v.trueHeading, \
                         'isFishing': v.isFishing }
        
        shipVInfoDicts.append(shipVInfoDict)
    
    return shipVInfoDicts
        

def convShipVInfosToPandaDF(shipVInfos):
    '''
        Convert a ship voyage information list into Pandas's data frame.
    '''
    
    # Convert a ship voyage information list into a dictionary list.
    shipVInfoDicts = convShipVInfosToDictList(shipVInfos)
    
    
    
    