from datetime import datetime
from dateutil import parser
from datetime import timedelta
import calendar

def convert_to_datetime(datetime_string):
    try:
        # print datetime_string
        return datetime.strptime(datetime_string, '%b %d, %Y')
    except ValueError:
        try :
            return datetime.strptime(datetime_string, '%Y-%m-%d')
        except ValueError:
            print 'error string is %s'%(datetime_string)

def convert_to_timestamp(datetime_string):
    try:
        # print datetime_string
        return calendar.timegm(datetime.strptime(datetime_string, '%b %d, %Y').utctimetuple())
    except ValueError:
        try :
            return calendar.timegm(datetime.strptime(datetime_string, '%Y-%m-%d').utctimetuple())
        except ValueError:
            print 'error string is %s'%(datetime_string)
def convert_iso_timestamp(timestamp):
    return parser.parse(timestamp)

def add_days_to_date(date, days):
    return date + timedelta(days = days)