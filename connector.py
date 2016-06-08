#coding=utf-8

import sys
import MySQLdb

reload(sys)

db_config = {
        'host': '172.18.50.11',
        'port': 3306,
        'user': 'crawler',
        'passwd': 'crawler1qaz',
        'db': 'crawler'
}

class Connect:
    def __init__(self):
        self.db = MySQLdb.connect(host=db_config['host'], port=db_config['port'],
                             user=db_config['user'], passwd=db_config['passwd'],
                             db=db_config['db'], charset='utf8', use_unicode=True)

if __name__ == '__main__':
    pass
