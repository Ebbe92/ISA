import os
from datetime import timedelta
basedir = os.path.abspath(os.path.dirname(__file__))
class Config(object):
    SECRET_KEY =  os.environ.get('SECRET_KEY') or "pTQX@L4j?7F7?zY8"
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL', '').replace(
        'postgres://', 'postgresql://') or \
        'sqlite:///' + os.path.join(basedir, 'app.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    REMEMBER_COOKIE_DURATION=timedelta(seconds=20)
    REMEMBER_COOKIE_REFRESH_EACH_REQUEST = True
    LOG_TO_STDOUT = os.environ.get('LOG_TO_STDOUT')
    
#'postgres://veddkabuqlvzdn:2be49a037e7474a47f4c4da9acd9819c64a8cf8a4e4d3d2324d66b4ec28dfbd5@ec2-54-228-32-29.eu-west-1.compute.amazonaws.com:5432/d7pkbkao09sikk'
 #MONGODB_SETTINGS = { 'db' : 'UTA_Enrollment'} #UTA_enrollment er navnet på databasen