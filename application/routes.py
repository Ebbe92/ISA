from application import app, db
from application.models import User, BMI
from application.forms import LoginForm
import numpy as np
from joblib import load
import pandas as pd
from flask import render_template, redirect, url_for, flash,get_flashed_messages
from flask import request
from flask_login import current_user, login_user, login_required, fresh_login_required
from werkzeug.urls import url_parse

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline
#import matplotlib.pyplot as plt

from matplotlib.figure import Figure
from matplotlib import style
import matplotlib.cm as cm
import matplotlib.colors as colors





from io import BytesIO
import base64
#import forms
# Categorical pipeline
categorical_preprocessing = Pipeline(
[
    ('Imputation', SimpleImputer(strategy='constant', fill_value='?')),
    ('One Hot Encoding', OneHotEncoder(handle_unknown='ignore')), #OneHotEncoder(handle_unknown='ignore')
]
)

# Numeric pipeline
numeric_preprocessing = Pipeline(
[
     ('Imputation', SimpleImputer(strategy='mean')),
     ('Scaling', StandardScaler()) #MinMax giver lidt mere accuracy - RobustScaler er god til at detektere overvægt (min. F1score på 0.8905)
]
)


# Creating preprocessing pipeline
preprocessing = make_column_transformer(
     (numeric_preprocessing, ['Age','Height','FCVC','NCP','CH2O','FAF','TUE']),
     (categorical_preprocessing, ['Gender','family_history_with_overweight','FAVC','CAEC','SCC','CALC','MTRANS']),
)

pipeline = Pipeline(
[('Preprocessing', preprocessing)]
)

@app.route("/")
@app.route("/index") #decorators - decorator for en funktion
@app.route("/home")
def index():
    return render_template("index.html", index=True)

@app.route('/login', methods=['GET', 'POST'])
def login():
    #error = None
    if current_user.is_authenticated:
       return redirect(url_for('project'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user is None or not user.check_password(form.password.data):
            flash('Invalid username or password')
            return redirect(url_for('login'))
        login_user(user)
        next_page = request.args.get('next')
        if not next_page or url_parse(next_page).netloc != '':
            next_page = url_for('index')
        return redirect(next_page)
    return render_template('login.html', title='Sign In', form=form)



@app.route("/project", methods =['GET', 'POST'])
@login_required
def project():
    if request.method == 'POST':
        data = pd.read_csv("ObesityData.csv")
        X = data.drop('NObeyesdad',axis=1)
        X = X.drop('SMOKE',axis=1)#prøvede at fjerne features for at gøre den mere præcist
        X = X.drop('Weight',axis=1)  
        t_list=(request.form.getlist('mycheckbox'))#returns a list
        t_list=convert(t_list) #convert all numbers in the list to float
        t_list_1=t_list[1:15]
        test_data=[]
        test_data.append(t_list_1)#appends the data - needs a 2d input
        t=pd.DataFrame(test_data, columns=['Gender', 'Age', 'Height', 'family_history_with_overweight','FAVC','FCVC','NCP','CAEC','CH2O', 'SCC','FAF','TUE','CALC','MTRANS'])
        if(len(X)>2111):
            X=X.drop(X.index[-1])
        X=X.append(t, ignore_index=True)
        X_fit2=pipeline.fit_transform(X)#transforming the data, so it's scaled according to the rest of the data
        F=[]
        F.append(X_fit2[-1])#appends the last list again - model needs 2d array
        model = load('model.joblib')
        y_pred_0 = model.predict(F)#predicts the result based on the latest input
        results=[]
        str_result= "At the moment the BMI value is {BMI}. If the user continues with this lifestyle, he'll/she'll be in the BMI-class {BMI_PRED}".format(BMI=BMI_result(t_list[0], t_list[3]), BMI_PRED=y_pred_0[0])
        results.append(str_result)#sends the results
        
        
      
        img=make_figure(t_list[3],t_list[0],y_pred_0[0])

        uid= current_user.id
        u=User.query.get(uid)
        b=u.posts.all()
        bmi_label_results=np.array([])
        bmi_predicted_results=np.array([])
        bmi_time_results=np.array([])
        for n in range(0, len(b)):
            date_time=b[n].timestamp
            d=date_time.strftime("%d %b, %Y") # https://www.programiz.com/python-programming/datetime/strftime
            bmi_label_results=np.append(bmi_label_results, b[n].BMI_label)
            bmi_predicted_results=np.append(bmi_predicted_results, b[n].BMI_predicted)
            bmi_time_results=np.append(bmi_time_results, d)
        f=np.array(list(zip(bmi_time_results, bmi_label_results, bmi_predicted_results)))
        final=np.array([])
        if len(b) > 5: 
          final=f[-5:, :]
          b_count=5
        else:
          final= f
          b_count=len(b)
        
        bi=BMI(BMI_label=str(BMI_result(t_list[0], t_list[3])), BMI_predicted=str(y_pred_0[0]), author=u)
        db.session.add(bi)
        db.session.commit()
       
        return render_template("result.html", index=True, results=results, img=img, b_count=b_count, f=final)
    else:
        return render_template("project.html", index=True)


def convert(float_str): 
  def is_float(s): 
    try: 
      float(s)
      return True
    except: 
      return False
  new_float=[]  
  for x in float_str:
      if is_float(x)==1:
          f=float(x)
          new_float.append(f)
      else:
          new_float.append(x)   
  return new_float

def BMI_calc(weight, height):
    bmi=((weight)/(height*height))
    return bmi
def BMI_result(weight, height):
    bmi_result=""
    bmi=BMI_calc(weight, height)
    print(bmi)
    if bmi<18.5:
        bmi_result ="Insufficient_Weight"
    elif bmi >= 18.5 and bmi <=24.9:
        bmi_result ="Normal_Weight"
    elif bmi >= 25 and bmi <=29.9:
        bmi_result ="Overweight"
    elif bmi >= 30.0 and bmi <=34.9:
        bmi_result ="Obesity_Type_I"
    elif bmi >= 35.0 and bmi <=39.9:
        bmi_result ="Obesity_Type_II"
    elif bmi >= 40:
        bmi_result ="Obesity_Type_III"
    return bmi_result

def make_figure(height, weight, bmi):
    style.use("ggplot")
    f= Figure((5,5))
    p = f.add_subplot()
    p.set_ylim(0,175)
    p.set_xlim(1.4,2.1)
    p.set_ylabel("Weight[kg]")
    p.set_xlabel("Height[meters]")
    def calc_weight (bmi, height):
        weight=bmi*(height*height)
        return weight
    h = np.arange(1.4,2.1,0.01) #GRAPH AS THRESHOLD
    p.plot(h, calc_weight(39.91,h), color ='red') #OBESITY III 
    p.plot(h,calc_weight(35,h)) #OBESITY II
    p.plot(h,calc_weight(30,h)) #OBESITY I
    p.plot(h,calc_weight(25,h)) #OVERWEIGHT
    p.plot(h,calc_weight(18.5,h), color='green')#NORMAL
    d= pd.read_csv("ObesityData.csv") #SCATTER PLOT OF THE DATA
    cond=[
        (d['NObeyesdad']=='Obesity_Type_III'),
        (d['NObeyesdad']=='Obesity_Type_II'),
        (d['NObeyesdad']=='Obesity_Type_I'),
        (d['NObeyesdad']=='Overweight_Level_II'),
        (d['NObeyesdad']=='Overweight_Level_I'),
        (d['NObeyesdad']=='Normal_Weight'),
        (d['NObeyesdad']=='Insufficient_Weight'),
    ]
    colorlist = ['red','orange','yellow','purple','purple','green','blue']
    d['c']=np.select(cond, colorlist)
    p.scatter(d['Height'], d['Weight'], c=d['c'].values)

 
  

    def weight_calc_bmi(bmi, height):
        bmi_val=0
        if bmi=='Insufficient_Weight':
            bmi_val=18.3
        elif bmi=='Normal_Weight':
            bmi_val=21.7 #median
        elif bmi=='Overweight_Level_I':
            bmi_val=26.25 #median
        elif bmi=='Overweight_Level_II':
            bmi_val=28.075 #median
        elif bmi=='Obesity_Type_I':
            bmi_val=32.45 #median
        elif bmi=='Obesity_Type_II':
            bmi_val=37.45 #median
        elif bmi=='Obesity_Type_III':
            bmi_val=42 #median

        w=bmi_val*(height*height)
        return w

    x=np.array([])
    x=np.append(x, height)
    x=np.append(x, height)
    y=np.array([])
    y=np.append(y,weight)
    y=np.append(y,weight_calc_bmi(bmi,height))
    c=np.array(["black", "white"])
    p.scatter(x,y, s=50, c=c)
    text=['BMI value at current time (BLACK DOT)', 'The BMI value that the model predicted(WHITE DOT)']
    
    p.annotate(
        text[0],
        (x[0],y[0]),
        xytext=(1.4, 160), 
        color='Black',
        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color ='black'))
    p.annotate(
        text[1],
        (x[1],y[1]),
        xytext=(1.4, 10), 
        color='Black',
        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color ='black')) #https://www.tutorialspoint.com/how-to-annotate-the-points-on-a-scatter-plot-with-automatically-placed-arrows-in-matplotlib


  
    # Save it to a temporary buffer. https://matplotlib.org/3.5.0/gallery/user_interfaces/web_application_server_sgskip.html
    buf = BytesIO()
    f.savefig(buf, format="jpeg")
    # Embed the result in the html output.
    data = base64.b64encode(buf.getbuffer()).decode("ascii") #https://docs.python.org/3/library/base64.html
    return data 




