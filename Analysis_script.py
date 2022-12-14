
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


# Create learning speed variable
performance_start = df_sessions[filters].groupby(['subject_uuid'])['performance_easy'].min().reset_index(name = 'performance_start') # Save the minimum performance for each subject
performance_end = df_sessions[filters].groupby(['subject_uuid'])['performance_easy'].max().reset_index(name = 'performance_end') # Save the maximum performance for each subject
learning_speed_df = pd.merge(performance_end, performance_start, on = "subject_uuid") # Merge the data frames with start and end performance for each subject
learning_speed_df = pd.merge(learning_speed_df, total_number_sessions, on = "subject_uuid") # Also save the total number of sessions to this data frame
learning_speed_df = pd.merge(learning_speed_df, total_number_training_days, on = "subject_uuid") # Also save the total number of training days to this data frame
learning_speed_df['learning_speed'] = (learning_speed_df['performance_end'] - learning_speed_df['performance_start'])/learning_speed_df['total_number_sessions'] # Create the training_speed variable by dividing the performance difference by the total number of sessions for each individual

# Create analysis dataframe with all variables created above
training_times_df = pd.merge(average_training_time, most_frequent_time, on = 'subject_uuid') # Merge the average training time and the most frequent training time ..
training_times_df = pd.merge(training_times_df, consistency_df, on = 'subject_uuid') # .. with the consistency per subject data frame

analysis_dataframe = pd.merge(learning_speed_df, training_times_df, on = 'subject_uuid') # Merge all variables created together into one data frame



#%%  

# STEP 3: ANALYSIS
from statsmodels.formula.api import ols

ols_model = ols(formula = 'learning_speed ~ time_consistency', data = analysis_dataframe)
ols_result = ols_model.fit()
ols_result.summary()
