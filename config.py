import os
from datetime import timedelta
basedir = os.path.abspath(os.path.dirname(__file__))
class Config(object):
    SECRET_KEY =  os.environ.get('SECRET_KEY') or "secret_string"
    #en special key - signature key - alt der bliver sendt til 
    #server ikke bliver altered eller hacked 
    #MONGODB_SETTINGS = { 'db' : 'UTA_Enrollment'} #UTA_enrollment er navnet på databasen
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL', '').replace(
        'postgres://', 'postgresql://') or \
        'sqlite:///' + os.path.join(basedir, 'app.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    REMEMBER_COOKIE_DURATION=timedelta(seconds=20)
    REMEMBER_COOKIE_REFRESH_EACH_REQUEST = True
    LOG_TO_STDOUT = os.environ.get('LOG_TO_STDOUT')
    