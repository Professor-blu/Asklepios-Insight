
from flask import Flask, request, jsonify, render_template, redirect, url_for, session
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from lifelines.utils import concordance_index
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField, SelectField, IntegerField
from wtforms.validators import DataRequired

Asklepios = Flask(__name__)
Asklepios.secret_key = 'your_secret_key'

#Load the models
with open('xgb.pkl', 'rb') as f:
    xgb_model = pickle.load(f)

with open('cph.pkl', 'rb') as f:
    cox_model = pickle.load(f)

#Load the scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

#Define the risk interventions
interventions = {
    0 : [
            'Regular Follow-ups (every 3-6 months)',
            'Health Education',
            'Support Groups',
            'Medication Refill Reminders'
    ],

    1 : [
            'Increased Frequency of Check-ups (every 1-3 months)',
            'Counseling Services',
            'Home Visits',
            'Enhanced Medication Reminders',
            'Nutritional Support',
            'Transportation Assistance'
    ],

    2 : [
            'Intensive Case Management',
            'Directly Observed Therapy (DOT)',
            'Mental Health Services',
            'Social Support Services',
            'Emergency Medical Services',
            'Customized Adherence Plans',
            'Intensive Nutritional Support',
            'Family Involvement'
    ]
}

#Dummy user data for authentication
users = {'admin': 'password'}

class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    remember_me = BooleanField('Remember Me')
    submit = SubmitField('Login')

class PatientDataForm(FlaskForm):
    cd4_first = IntegerField('CD4 count at first visit', validators=[DataRequired()])
    viral_first = IntegerField('Viral load at first visit', validators=[DataRequired()])
    cd4_recent = IntegerField('CD4 count at most recent visit', validators=[DataRequired()])
    viral_recent = IntegerField('Viral load at most recent visit', validators=[DataRequired()])
    age_first = IntegerField('Age at first visit', validators=[DataRequired()])
    missed_doses = IntegerField('Missed doses in the last month', validators=[DataRequired()])
    duration_followups = IntegerField('Duration of followups', validators=[DataRequired()])
    duration_hiv_positive = IntegerField('Duration HIV Positive', validators=[DataRequired()])
    gender = SelectField('Gender', choices=[('Male', 'Male'), ('Female', 'Female')], validators=[DataRequired()])
    previous_art_exposure = SelectField('Previous ART exposure', choices=[('Yes', 'Yes'), ('No', 'No')], validators=[DataRequired()])
    art_regimen = SelectField('Current ART regimen', choices=[('Regimen A', 'Regimen A'), ('Regimen B', 'Regimen B'), ('Regimen C', 'Regimen C')], validators=[DataRequired()])
    employment_status = SelectField('Employment status', choices=[('Employed', 'Employed'), ('Other', 'Other'), ('Retired', 'Retired'), ('Student', 'Student'), ('Unemployed', 'Unemployed')], validators=[DataRequired()])
    education_level = SelectField('Education level', choices=[('Primary education', 'Primary education'), ('Secondary education', 'Secondary education'), ('Tertiary education', 'Tertiary education')], validators=[DataRequired()])
    income_level = SelectField('Income level', choices=[('Low', 'Low'), ('Medium', 'Medium'), ('High', 'High'), ('Prefer not to say', 'Prefer not to say')], validators=[DataRequired()])
    marital_status = SelectField('Marital status', choices=[('Married', 'Married'), ('Single', 'Single'), ('Widowed', 'Widowed'), ('Other', 'Other')], validators=[DataRequired()])
    substance_use = SelectField('Substance use history', choices=[('None', 'None'), ('Illicit drugs', 'Illicit drugs'), ('Tobacco', 'Tobacco'), ('Other', 'Other')], validators=[DataRequired()])
    comorbidities = SelectField('Comorbidities', choices=[('None', 'None'), ('Diabetes', 'Diabetes'), ('Hepatitis B/C', 'Hepatitis B/C'), ('Hypertension', 'Hypertension'), ('Tuberculosis', 'Tuberculosis'), ('Other', 'Other')], validators=[DataRequired()])
    reported_symptoms = SelectField('Reported symptoms', choices=[('None', 'None'), ('Fever', 'Fever'), ('Night sweats', 'Night sweats'), ('Weight loss', 'Weight loss'), ('Other', 'Other')], validators=[DataRequired()])
    dietary_habits = SelectField('Dietary habits', choices=[('Healthy', 'Healthy'), ('Poor', 'Poor')], validators=[DataRequired()])
    physical_activity = SelectField('Physical activity', choices=[('Regular', 'Regular'), ('Irregular', 'Irregular')], validators=[DataRequired()])
    adherence_to_art = SelectField('Adherence to ART', choices=[('Never', 'Never'), ('Rarely', 'Rarely'), ('Sometimes', 'Sometimes'), ('Often', 'Often')], validators=[DataRequired()])
    adverse_event = SelectField('Adverse event', choices=[('Yes', 'Yes'), ('No', 'No')], validators=[DataRequired()])
    submit = SubmitField('Submit')

@Asklepios.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        username = form.username.data
        password = form.password.data
        if username in users and users[username] == password:
            session['username'] = username
            return redirect(url_for('home'))
        else:
            return render_template('login.html', form=form, error='Invalid username or password')
    return render_template('login.html', form=form)


@Asklepios.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))


@Asklepios.route('/')
def home():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('index.html', form=PatientDataForm())


@Asklepios.route('/predict', methods=['POST'])
def predict():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    form = PatientDataForm()
    if form.validate_on_submit():
        patient_data = pd.DataFrame([{
            'CD4 count at first visit': form.cd4_first.data,
            'Viral load at first visit': form.viral_first.data,
            'CD4 count at most recent visit': form.cd4_recent.data,
            'Viral load at most recent visit': form.viral_recent.data,
            'Age at first visit': form.age_first.data,
            'Missed doses in the last month': form.missed_doses.data,
            'Duration of followups': form.duration_followups.data,
            'DurationHIVPositive': form.duration_hiv_positive.data,
            'Gender_Male': int(form.gender.data == 'Male'),
            'Previous ART exposure_Yes': int(form.previous_art_exposure.data == 'Yes'),
            'Current ART regimen_Regimen B': int(form.art_regimen.data == 'Regimen B'),
            'Current ART regimen_Regimen C': int(form.art_regimen.data == 'Regimen C'),
            'Employment status_Other': int(form.employment_status.data == 'Other'),
            'Employment status_Retired': int(form.employment_status.data == 'Retired'),
            'Employment status_Student': int(form.employment_status.data == 'Student'),
            'Employment status_Unemployed': int(form.employment_status.data == 'Unemployed'),
            'Education level_Primary education': int(form.education_level.data == 'Primary education'),
            'Education level_Secondary education': int(form.education_level.data == 'Secondary education'),
            'Education level_Tertiary education': int(form.education_level.data == 'Tertiary education'),
            'Income level_Low': int(form.income_level.data == 'Low'),
            'Income level_Medium': int(form.income_level.data == 'Medium'),
            'Income level_Prefer not to say': int(form.income_level.data == 'Prefer not to say'),
            'Marital status_Married': int(form.marital_status.data == 'Married'),
            'Marital status_Other': int(form.marital_status.data == 'Other'),
            'Marital status_Single': int(form.marital_status.data == 'Single'),
            'Marital status_Widowed': int(form.marital_status.data == 'Widowed'),
            'Substance use history_Illicit drugs': int(form.substance_use.data == 'Illicit drugs'),
            'Substance use history_Other': int(form.substance_use.data == 'Other'),
            'Substance use history_Tobacco': int(form.substance_use.data == 'Tobacco'),
            'Comorbidities_Diabetes': int(form.comorbidities.data == 'Diabetes'),
            'Comorbidities_Hepatitis B/C': int(form.comorbidities.data == 'Hepatitis B/C'),
            'Comorbidities_Hypertension': int(form.comorbidities.data == 'Hypertension'),
            'Comorbidities_None': int(form.comorbidities.data == 'None'),
            'Comorbidities_Other': int(form.comorbidities.data == 'Other'),
            'Comorbidities_Tuberculosis': int(form.comorbidities.data == 'Tuberculosis'),
            'Reported symptoms_Fever': int(form.reported_symptoms.data == 'Fever'),
            'Reported symptoms_Night sweats': int(form.reported_symptoms.data == 'Night sweats'),
            'Reported symptoms_None': int(form.reported_symptoms.data == 'None'),
            'Reported symptoms_Other': int(form.reported_symptoms.data == 'Other'),
            'Reported symptoms_Weight loss': int(form.reported_symptoms.data == 'Weight loss'),
            'Dietary habits_Healthy': int(form.dietary_habits.data == 'Healthy'),
            'Dietary habits_Poor': int(form.dietary_habits.data == 'Poor'),
            'Physical activity_Irregular': int(form.physical_activity.data == 'Irregular'),
            'Physical activity_Regular': int(form.physical_activity.data == 'Regular'),
            'Adherence to ART_Never': int(form.adherence_to_art.data == 'Never'),
            'Adherence to ART_Often': int(form.adherence_to_art.data == 'Often'),
            'Adherence to ART_Rarely': int(form.adherence_to_art.data == 'Rarely'),
            'Adherence to ART_Sometimes': int(form.adherence_to_art.data == 'Sometimes'),
            'Adverse event_Yes': int(form.adverse_event.data == 'Yes'),
            'Date of exit from the study': form.exit_date.data,
            'Start date of current ART': form.start_date_art.data,
            'Date of most recent visit': form.recent_visit_date.data,
            'Date confirmed HIV positive': form.hiv_positive_date.data
        }])

        #Calculate DurationToExitDate and DurationHIVPositive
        patient_data['DurationTOExitDate'] = (pd.to_datetime(patient_data['Date of exit from the study']) - pd.to_datetime(patient_data['Start date of current ART'])).dt.days
        patient_data['DurationHIVPositive'] = (pd.to_datetime(patient_data['Date of most recent visit']) - pd.to_datetime(patient_data['Date confirmed HIV positive'])).dt.days

        #Drop the date columns as they are no longer needed
        patient_data = patient_data.drop(['Date of exit from study', 'Start date of current ART', 'Date of most recent visit', 'Date confirmed HIV positive'])

        # Convert negative values to zero
        patient_data = convert_to_zero(patient_data, 'DurationToExitDate')
        patient_data = convert_to_zero(patient_data, 'DurationHIVPositive')

        # Calculate risk score
        patient_data = calculate_risk_score(patient_data)

        # Normalize the Risk Score to a range of 0 to 100
        patient_data['Risk Score'] = patient_data['Risk Score'].apply(lambda x: (x - patient_data['Risk Score'].min()) / (patient_data['Risk Score'].max() - patient_data['Risk Score'].min()) * 100)

        # Categorize the risk
        patient_data['Risk Category'] = patient_data['Risk Score'].apply(categorize_risk)

        # Encode risk categories
        category_mapping = {'low risk': 0, 'medium risk': 1, 'high risk': 2}
        patient_data['Risk Category'] = patient_data['Risk Category'].map(category_mapping)

        # Standardize the patient data
        patient_data_scaled = scaler.transform(patient_data)

        # Predict risk category and probability using the XGBoost classifier
        risk_category = xgb_model.predict(patient_data_scaled)
        risk_probability = xgb_model.predict_proba(patient_data_scaled).max(axis=1)[0]

        # Map the encoded risk category back to the original labels
        risk_category_mapping = {0: 'low risk', 1: 'medium risk', 2: 'high risk'}
        risk_category = risk_category_mapping[risk_category[0]]

        # Predict survival time using the CoxPH model
        survival_time_prediction = cox_model.predict(patient_data_scaled)[0]

        # Determine the intervention based on the risk category
        intervention = interventions[risk_category[0]]

        # Prepare the result dictionary
        result = {
            'risk_category': risk_category,
            'risk_probability': risk_probability,
            'risk_score': patient_data['Risk Score'][0],
            'predicted_survival_time': survival_time_prediction,
            'intervention': intervention
        }

        return render_template('result.html', result=result)

    return redirect(url_for('home'))

def calculate_risk_score(df):
    def assign_risk(row):
        # Age at First Visit
        if row['Age at first visit'] < 30:
            age_risk = 0
        elif 30 <= row['Age at first visit'] < 40:
            age_risk = 1
        elif 40 <= row['Age at first visit'] < 50:
            age_risk = 2
        elif 50 <= row['Age at first visit'] < 60:
            age_risk = 3
        elif 60 <= row['Age at first visit'] < 70:
            age_risk = 4
        else:
            age_risk = 5

        # CD4 Count at Most Recent Visit
        if row['CD4 count at most recent visit'] > 500:
            cd4_risk = 0
        elif 350 <= row['CD4 count at most recent visit'] <= 499:
            cd4_risk = 1
        elif 200 <= row['CD4 count at most recent visit'] <= 349:
            cd4_risk = 3
        else:
            cd4_risk = 5

        # Viral Load at Most Recent Visit
        if row['Viral load at most recent visit'] < 20:
            viral_load_risk = 0
        elif 20 <= row['Viral load at most recent visit'] <= 1000:
            viral_load_risk = 2
        else:
            viral_load_risk = 5

        # Adherence to ART
        adherence_risk_map = {
            'Always': 0,
            'Often': 1,
            'Sometimes': 3,
            'Rarely': 4,
            'Never': 5
        }
        adherence_risk = adherence_risk_map.get(row['Adherence to ART'], 0)

        # Comorbidities
        comorbidities_risk_map = {
            'None': 0,
            'Diabetes': 2,
            'Hypertension': 2,
            'Tuberculosis': 2,
            'Hepatitis B/C': 2,
            'Cardiovascular diseases': 2,
            'Other': 2
        }
        
        # Handle NaN values in comorbidities
        if pd.isna(row['Comorbidities']):
            comorbidities_risk = 0
        else:
            comorbidities_risk = sum([comorbidities_risk_map.get(comorbidity, 2) for comorbidity in row['Comorbidities'].split(', ')])

        if comorbidities_risk > 2:
            comorbidities_risk = 5

        # Substance Use History
        substance_use_risk_map = {
            'None': 0,
            'Alcohol': 3,
            'Tobacco': 3,
            'Illicit drugs': 5,
            'Other': 5
        }
        substance_use_risk = substance_use_risk_map.get(row['Substance use history'], 0)

        # Calculate total risk score
        total_risk_score = age_risk + cd4_risk + viral_load_risk + adherence_risk + comorbidities_risk + substance_use_risk

        return total_risk_score

    # Apply the function to each row
    df['Risk Score'] = df.apply(assign_risk, axis=1)
    return df

def convert_to_zero(df, column_name):
    """
    This function converts all negative values in a given column to zero.

    Args:
        df (pandas.DataFrame): The dataframe containing the column to be processed.
        column_name (str): The name of the column to be processed.

    Returns:
        pandas.DataFrame: The modified dataframe with negative values converted to zero.
    """
    df[column_name] = df[column_name].clip(lower=0)
    return df
def categorize_risk(score):
    low_risk_threshold = 33.33
    medium_risk_threshold = 66.67
    if score <= low_risk_threshold:
        return 'low risk'
    elif score <= medium_risk_threshold:
        return 'medium risk'
    else:
        return 'high risk'

if __name__ == '__main__':
    Asklepios.run(debug=True)
        