from datetime import datetime

def convertDate(indate):
    return datetime.strptime(indate[:-7] + '0000' , '%Y-%m-%dT%H:%M:%S.%f')
