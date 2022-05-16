from application import db
from application import login
#from manage import db,app
from flask_login import UserMixin
from datetime import datetime, timedelta
from werkzeug.security import generate_password_hash, check_password_hash

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), index=True, unique=True)
    email = db.Column(db.String(120), index=True, unique=True)
    password_hash = db.Column(db.String(128))
    posts = db.relationship('BMI', backref='author', lazy='dynamic')

    
    def __repr__(self):
        return '<User {}>'.format(self.username)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class BMI(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    BMI_label = db.Column(db.String(140))
    BMI_predicted = db.Column(db.String(140))
    timestamp = db.Column(db.DateTime, index=True, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))

    def __repr__(self):
        return '<BMI {BMI_l},  {BMI_p}, {time}>'.format(BMI_l=self.BMI_label, BMI_p=self.BMI_predicted, time=self.timestamp)


@login.user_loader
def load_user(id):
    return User.query.get(int(id))