
# This script is created to analyse the IBL behavioural data
# We investigate whether the light cycle and training time consistency affect learning performance


#%%  

# STEP 0: SET UP

# Load packages
import datajoint as dj
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns


#%%  

# STEP 1: EXPLORATION AND SELECTION




#%%  

# STEP 2: CREATE VARIABLES


# Inspect the different columns in the subjects data frame
df_subjects.columns


# Create a light_cycle variable per lab [note: values 1 and 0 are randomly divided for now as it is a blinded analysis]
light_cycle = np.array([1,0,1,0,1,0,1,0,1,0]) # Randomly divided values of one and zero to a light_cycle array
lab_names = df_subjects['lab_name'].unique() # Save the lab_names to a vector
light_cycle_dataframe = pd.DataFrame({'light_cycle':[1,0,1,0,1,0,1,0,1,0],
                                       'lab_name': ['angelakilab', 'churchlandlab', 'churchlandlab_ucla', 'cortexlab',
                                              'danlab', 'hoferlab', 'mainenlab', 'mrsicflogellab', 'wittenlab','zadorlab']}) # Merge the light cycle values to the lab_names
df_subjects = pd.merge(df_subjects, light_cycle_dataframe, on = 'lab_name') # Save the light cycle values to the subjects data frame
df_subjects.columns # Inspect the columns again (check whether the merging worked and whether each subject has a light cycle value)
subjects_light_cycle = df_subjects[['subject_uuid','light_cycle']]


# Investigate the subject id's 
df_subjects['subject_uuid'] # 233 unique subject uuid's in the subjects dataframe
len(df_sessions['subject_uuid'].unique()) # 197 unique subject uuid's


# Create a total number of sessions per subject variable
total_number_sessions = df_sessions.groupby(['subject_uuid'])['session_start_time'].count().reset_index(name = 'total_number_sessions') # Count the total number of sessions for each subject


# Create a total number of training days variable
total_number_training_days = df_sessions.groupby(['subject_uuid'])['session_dates'].count().reset_index(name = 'total_number_training_days') # Count the total number of training days for each subject


# Create a session number per subject variable 
List_subject_uuids = df_sessions['subject_uuid'].unique() # Create a list with all unique subject_uuid's in the sessions data frame
filters = df_sessions["subject_uuid"].isin(List_subject_uuids) # Use this list to set the filters

df_sessions.loc[filters, "session_number"] = (
    df_sessions[filters].groupby(["subject_uuid"])["session_start_time"].transform(lambda x: pd.CategoricalIndex(x).codes + 1)) # Save the session number for each subject


# Create a training day per subject variable
df_sessions.loc[filters, "training_day"] = (
    df_sessions[filters].groupby(["subject_uuid"])["session_dates"].transform(lambda x: pd.CategoricalIndex(x).codes + 1)) # Save the training day for each subject


# Create an average training time per subject variable
average_training_time = df_sessions[filters].groupby("subject_uuid")["session_time_hour"].mean().reset_index(name = "average_training_time") # Save the average training time for each subject


# Create a most common training time per subject variable
time_frequency = df_sessions[filters].groupby("subject_uuid")["session_time_hour"].value_counts().reset_index(name = 'frequency') # Count the occurences of the session_time_hour
most_frequent_time = time_frequency.groupby("subject_uuid").max().reset_index() # Save the maximum frequency of the session_time_hour
most_frequent_time = most_frequent_time.iloc[:,0:2] # Only save the subject_uuid and session_time_hour
most_frequent_time.rename(columns = {'session_time_hour':'most_frequent_time'}, inplace = True) # Change the column name to most_frequent_time


# Create consistency training time variable
max_per_subject = time_frequency[filters].groupby(['subject_uuid'])['frequency'].max().reset_index(name = 'max') # Save the maximum frequency of the training time for each subject
sum_per_subject = time_frequency[filters].groupby(['subject_uuid'])['frequency'].sum().reset_index(name = 'sum') # Save the total frequency of the training time for each subject
consistency_df = pd.merge(max_per_subject, sum_per_subject, on = 'subject_uuid') # Merge the two data frames together
consistency_df['time_consistency'] = consistency_df['max']/consistency_df['sum'] # Use the maximum frequency and the total frequency of the training time to calculate the consistency in training time for each subject



# Create different learning_speed variables

# First version (speed by performance high-low/total sessions)
performance_start = df_sessions[filters].groupby(['subject_uuid'])['performance_easy'].min().reset_index(name = 'performance_start') # Save the minimum performance for each subject
performance_end = df_sessions[filters].groupby(['subject_uuid'])['performance_easy'].max().reset_index(name = 'performance_end') # Save the maximum performance for each subject

learning_speed_df = pd.merge(performance_end, performance_start, on = "subject_uuid") # Merge the data frames with start and end performance for each subject
learning_speed_df = pd.merge(learning_speed_df, total_number_sessions, on = "subject_uuid") # Also save the total number of sessions to this data frame
learning_speed_df = pd.merge(learning_speed_df, total_number_training_days, on = "subject_uuid") # Also save the total number of training days to this data frame
learning_speed_df['learning_speed_1'] = (learning_speed_df['performance_end'] - learning_speed_df['performance_start'])/learning_speed_df['total_number_sessions'] # Create the training_speed variable by dividing the performance difference by the total number of sessions for each individual


# Second version (speed by performance last_day-first_day/total sessions)
session_start = df_sessions[filters].groupby(['subject_uuid'])['session_number'].min().reset_index(name = 'session_number') # Save the row of the first session per subject
session_end = df_sessions.groupby(['subject_uuid'])['session_number'].max().reset_index(name = 'session_number') # Save the row of the last session per subject

subject_session_performance = df_sessions[['subject_uuid','session_number','performance_easy' ]] # Create new dataframe with only the subject id, the session number and the performance
performance_session_start = pd.merge(session_start, subject_session_performance, how = 'inner', on = ['subject_uuid', 'session_number']) # Merge dataframes with only the first session to save the performance at the first session
performance_session_end = pd.merge(session_end, subject_session_performance, how = 'inner', on = ['subject_uuid', 'session_number']) # Merge dataframes with only the last session to save the performance at the last session
performance_session_start = performance_session_start[['subject_uuid', 'performance_easy']] # Remove the session number
performance_session_end = performance_session_end [['subject_uuid', 'performance_easy']] # Remove the session number
performance_session_start = performance_session_start.rename(columns = {'performance_easy': 'performance_session_start'}) # Change the colname
performance_session_end = performance_session_end.rename(columns = {'performance_easy': 'performance_session_end'}) # Change the colname
                                          
performance_sessions = pd.merge(performance_session_start, performance_session_end, on = 'subject_uuid') # Merge the performance at the start and the performance at the end
learning_speed_df = pd.merge(learning_speed_df, performance_sessions, on = "subject_uuid") # Merge the performance start and end to the learning speed data frame
learning_speed_df['learning_speed_2'] = (learning_speed_df['performance_session_end'] - learning_speed_df['performance_session_start'])/learning_speed_df['total_number_sessions'] # Create the training_speed variable by dividing the performance difference by the total number of sessions for each individual


# Third version (sessions to trained)
trained_animals = df_sessions.loc[(df_sessions['training_status'] == "trained_1a") | (df_sessions['training_status'] == "trained_1b")] # Save the rows where the training_status is trained_1a or trained_1b
trained_animals_only = trained_animals.groupby('subject_uuid')['session_number'].min().reset_index(name = 'session_to_trained') # Save the minimal session_number per subject 
learning_speed_df = pd.merge(learning_speed_df, trained_animals_only, on ='subject_uuid') # Merge the session_to_trained variable with the learning_speed_df
learning_speed_df = learning_speed_df.drop_duplicates() # Remove duplicate observations


# Fourth version (sessions to 80% of performance easy)
performance_end[['eighty_performance_end']] = performance_end[['performance_end']]*0.8 # Save 80% of the performance
performence_eight = performance_end[['subject_uuid', 'eighty_performance_end']] # Save the value in a new dataframe
df_sessions = pd.merge(df_sessions, performance_end, on = 'subject_uuid') # Merge this value to the sessions dataframe
index_eighty = np.where(df_sessions['performance_easy'] >= df_sessions['eighty_performance_end'] ) # Find where the performance is similar or greater than 80% for each individual
df_sessions_eighty = df_sessions.iloc[index_eighty[0]] # Save these rows
df_sessions_eighty_over = df_sessions_eighty.groupby('subject_uuid')['session_number'].min().reset_index(name = 'sessions_to_eighty') # Get the session number for the observations where the performance is greater than 80%
df_sessions_eighty_over = df_sessions_eighty_over[['subject_uuid', 'sessions_to_eighty']] # Only save the number to the data frame
learning_speed_df = pd.merge(learning_speed_df, df_sessions_eighty_over, on = "subject_uuid") # Merge this to the dataframe



#%%  

# Analyse the different learning speed variables
sns.pairplot(learning_speed_df[['learning_speed_1', 'learning_speed_2', 'session_to_trained', 'sessions_to_eighty']])
plt.show()


#%%  
# Create analysis dataframe with all variables created above
training_times_df = pd.merge(average_training_time, most_frequent_time, on = 'subject_uuid') # Merge the average training time and the most frequent training time ..
training_times_df = pd.merge(training_times_df, consistency_df, on = 'subject_uuid') # .. with the consistency per subject data frame

analysis_dataframe = pd.merge(learning_speed_df, training_times_df, on = 'subject_uuid') # Merge all variables created together into one data frame
analysis_dataframe = pd.merge(analysis_dataframe, subjects_light_cycle, on = 'subject_uuid')
analysis_dataframe.columns


#%%  

# STEP 3: ANALYSIS
from statsmodels.formula.api import ols

ols_model = ols(formula = 'learning_speed ~ time_consistency + light_cycle', data = analysis_dataframe)
ols_result = ols_model.fit()
ols_result.summary()


ols_model_all = ols(formula = 'learning_speed ~ performance_end + performance_start + total_number_sessions + total_number_training_days + average_training_time + most_frequent_time + max + sum + time_consistency + light_cycle',  data = analysis_dataframe)
ols_result_all = ols_model.fit()
ols_result_all.summary()



