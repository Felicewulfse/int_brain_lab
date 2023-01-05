
# Felice Wulfse
# January 2023

# This script is created to analyse the IBL behavioural data
# We investigate whether the light cycle and training time consistency affect learning performance


#%%  

# STEP 0: SET UP AND LOAD DATA
# %reset -f # Remove list

# Load packages
import datajoint as dj
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import statistics 


# Read in data
df_subjects = pd.read_excel('/Users/felicewulfse/Library/Mobile Documents/com~apple~CloudDocs/Documents/Universiteit/MSc Applied Cognitive Psychology/Thesis/My_repo_int_brain_lab/df_subjects.xlsx')
df_sessions = pd.read_excel('/Users/felicewulfse/Library/Mobile Documents/com~apple~CloudDocs/Documents/Universiteit/MSc Applied Cognitive Psychology/Thesis/My_repo_int_brain_lab/df_sessions.xlsx')


#%%  

# STEP 1: EXPLORATION AND SELECTION

# Filter out the observations that are not representative due to errors
        # //



#%%  

# STEP 2: CREATE VARIABLES NEEDED FOR THE ANALYSIS

# Inspect the different columns in the subjects data frame
df_subjects.columns
df_sessions.columns


# Investigate the subject id's 
df_subjects['subject_uuid'] # 233 unique subject uuid's in the subjects dataframe
len(df_sessions['subject_uuid'].unique()) # 197 unique subject uuid's


# Create the session_date, session_time and session_time hour variable
df_sessions['session_dates'] = pd.to_datetime(df_sessions['session_start_time']).dt.date
df_sessions['session_time'] = pd.to_datetime(df_sessions['session_start_time']).dt.time
df_sessions['session_time_hour'] = df_sessions['session_time'].apply(lambda x:x.hour)


# Create a light_cycle variable per lab [note: values 1 and 0 are randomly divided for now as it is a blinded analysis]
light_cycle = np.array([1,0,1,0,1,0,1,0,1,0]) # Randomly divided values of one and zero to a light_cycle array
lab_names = df_subjects['lab_name'].unique() # Save the lab_names to a vector
light_cycle_dataframe = pd.DataFrame({'light_cycle':[1,0,1,0,1,0,1,0,1,0],
                                       'lab_name': ['angelakilab', 'churchlandlab', 'churchlandlab_ucla', 'cortexlab',
                                              'danlab', 'hoferlab', 'mainenlab', 'mrsicflogellab', 'wittenlab','zadorlab']}) # Merge the light cycle values to the lab_names
df_subjects = pd.merge(df_subjects, light_cycle_dataframe, on = 'lab_name') # Save the light cycle values to the subjects data frame
subjects_light_cycle = df_subjects[['subject_uuid','lab_name','light_cycle']]
df_sessions = pd.merge(df_sessions, subjects_light_cycle, on = 'subject_uuid') # Save the light cycle values to the sessions data frame


# Create a total_number_sessions and total_number_training_days varaible per subject variable
total_number_sessions = df_sessions.groupby(['subject_uuid'])['session_start_time'].count().reset_index(name = 'total_number_sessions') # Count the total number of sessions for each subject
total_number_training_days = df_sessions.groupby(['subject_uuid'])['session_dates'].count().reset_index(name = 'total_number_training_days') # Count the total number of training days for each subject


# Save the session_number and training_day variable for the session data frame
List_subject_uuids = df_sessions['subject_uuid'].unique() # Create a list with all unique subject_uuid's in the sessions data frame
filters = df_sessions["subject_uuid"].isin(List_subject_uuids) 
df_sessions.loc[filters,"session_number"] = (
    df_sessions.groupby([filters,"subject_uuid"])["session_start_time"].transform(lambda x: pd.CategoricalIndex(x).codes + 1)) # Save the session number for each subject
df_sessions.loc["training_day"] = (
    df_sessions.groupby([filters,"subject_uuid"])["session_dates"].transform(lambda x: pd.CategoricalIndex(x).codes + 1)) # Save the training day for each subject


# Create an average training time per subject variable
average_training_time = df_sessions.groupby("subject_uuid")["session_time_hour"].mean().reset_index(name = "average_training_time") # Save the average training time for each subject


# Create a most common training time per subject variable
time_frequency = df_sessions.groupby("subject_uuid")["session_time_hour"].value_counts().reset_index(name = 'frequency') # Count the occurences of the session_time_hour
most_frequent_time = time_frequency.groupby("subject_uuid").max().reset_index() # Save the maximum frequency of the session_time_hour
most_frequent_time = most_frequent_time.iloc[:,0:2] # Only save the subject_uuid and session_time_hour
most_frequent_time.rename(columns = {'session_time_hour':'most_frequent_time'}, inplace = True) # Change the column name to most_frequent_time



## CREATE CONSISTENCY TRAINING TIME VARIABLES ##

# 1: Consistency training time, defined as the max_time_frequency divided by the sum_time_frequency
max_per_subject = time_frequency.groupby(['subject_uuid'])['frequency'].max().reset_index(name = 'max') # Save the maximum frequency of the training time for each subject
sum_per_subject = time_frequency.groupby(['subject_uuid'])['frequency'].sum().reset_index(name = 'sum') # Save the total frequency of the training time for each subject
consistency_df = pd.merge(max_per_subject, sum_per_subject, on = 'subject_uuid') # Merge the two data frames together
consistency_df['time_consistency_1'] = consistency_df['max']/consistency_df['sum'] # Use the maximum frequency and the total frequency of the training time to calculate the consistency in training time for each subject


# 2: Consistency training time, defined as the standard deviation of the session_time_hour distribution
df_sessions_time = df_sessions[['subject_uuid', 'session_time_hour']] # Create a new data frame with only the subject_uuid and the session_time_hour
sd_time_hour = df_sessions_time.groupby(['subject_uuid'])['session_time_hour'].std().reset_index(name = 'time_consistency_2')
sd_time_hour = pd.DataFrame(sd_time_hour) # Save the values as a dataframe
sd_time_hour[['reverse_time_consistency_2']] = 1/(sd_time_hour[['time_consistency_2']]) # ^-1 to reverse the time consistency definition
consistency_df = pd.merge(consistency_df, sd_time_hour, on = 'subject_uuid') # Merge the two data frames together

# Inspect the two different definitions
consistency_df.columns # Check columns
np.corrcoef(consistency_df['time_consistency_1'], consistency_df['time_consistency_2']) # Calculate the correlation
 


## CREATE LEARNING SPEED VARIABLES ##

# 1: Learning speed, defined as the max_performance minus the min_performance divided by the total number of sessions
performance_start = df_sessions.groupby(['subject_uuid'])['performance_easy'].min().reset_index(name = 'performance_start') # Save the minimum performance for each subject
performance_end = df_sessions.groupby(['subject_uuid'])['performance_easy'].max().reset_index(name = 'performance_end') # Save the maximum performance for each subject

learning_speed_df = pd.merge(performance_end, performance_start, on = "subject_uuid") # Merge the data frames with start and end performance for each subject
learning_speed_df = pd.merge(learning_speed_df, total_number_sessions, on = "subject_uuid") # Also save the total number of sessions to this data frame
learning_speed_df = pd.merge(learning_speed_df, total_number_training_days, on = "subject_uuid") # Also save the total number of training days to this data frame
learning_speed_df['learning_speed_1'] = (learning_speed_df['performance_end'] - learning_speed_df['performance_start'])/learning_speed_df['total_number_sessions'] # Create the training_speed variable by dividing the performance difference by the total number of sessions for each individual


# 2: Learning speed, defined as the performance at the last_day minus the performance at the first_day divided by the total number of sessions
session_start = df_sessions.groupby(['subject_uuid'])['session_number'].min().reset_index(name = 'session_number') # Save the row of the first session per subject
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


# 3: Learning speed, defined by the number of sessions untill achieved trained_1a/ trained_1b
trained_animals = df_sessions.loc[(df_sessions['training_status'] == "trained_1a") | (df_sessions['training_status'] == "trained_1b")] # Save the rows where the training_status is trained_1a or trained_1b
trained_animals_only = trained_animals.groupby('subject_uuid')['session_number'].min().reset_index(name = 'sessions_to_trained') # Save the minimal session_number per subject 
trained_animals_only[['reverse_sessions_to_trained']] = 1/(trained_animals_only[['sessions_to_trained']]) # ^-1 to reverse the learning speed definition
learning_speed_df = pd.merge(learning_speed_df, trained_animals_only, on ='subject_uuid') # Merge the sessions_to_trained variable with the learning_speed_df
learning_speed_df = learning_speed_df.drop_duplicates() # Remove duplicate observations


# 4: Learning speed, defined by the number of sessions untill achieved 80% of max performance_easy
performance_end[['eighty_performance_end']] = performance_end[['performance_end']]*0.8 # Save 80% of the performance
performence_eight = performance_end[['subject_uuid', 'eighty_performance_end']] # Save the value in a new dataframe
df_sessions = pd.merge(df_sessions, performance_end, on = 'subject_uuid') # Merge this value to the sessions dataframe
index_eighty = np.where(df_sessions['performance_easy'] >= df_sessions['eighty_performance_end']) # Find where the performance is similar or greater than 80% for each individual
df_sessions_eighty = df_sessions.iloc[index_eighty[0]] # Save these rows
df_sessions_eighty_over = df_sessions_eighty.groupby('subject_uuid')['session_number'].min().reset_index(name = 'sessions_to_eighty') # Get the session number for the observations where the performance is greater than 80%
df_sessions_eighty_over = df_sessions_eighty_over[['subject_uuid', 'sessions_to_eighty']] # Only save the number to the data frame
df_sessions_eighty_over[['reverse_sessions_to_eighty']] = 1/(df_sessions_eighty_over[['sessions_to_eighty']]) # ^-1 to reverse the learning speed definition
learning_speed_df = pd.merge(learning_speed_df, df_sessions_eighty_over, on = "subject_uuid") # Merge this to the dataframe


# Inspect the four different definitions
learning_speed_df.columns # Check the columns
np.corrcoef(learning_speed_df['learning_speed_1'], learning_speed_df['learning_speed_2'])  # Calculate the correlations
np.corrcoef(learning_speed_df['sessions_to_trained'], learning_speed_df['sessions_to_eighty'])

np.corrcoef(learning_speed_df['learning_speed_1'], learning_speed_df['sessions_to_trained'])
np.corrcoef(learning_speed_df['learning_speed_1'], learning_speed_df['sessions_to_eighty'])
np.corrcoef(learning_speed_df['learning_speed_2'], learning_speed_df['sessions_to_trained'])
np.corrcoef(learning_speed_df['learning_speed_2'], learning_speed_df['sessions_to_eighty'])



#%%   

## DUMP ## 

### Check this filters part later

# Create a session number per subject variable 
List_subject_uuids = df_sessions['subject_uuid'].unique() # Create a list with all unique subject_uuid's in the sessions data frame
filters = df_sessions["subject_uuid"].isin(List_subject_uuids) # Use this list to set the filters

###


#%%  

# Create plots to visualise the different variables

# Add a variable to highlight an example mouse to the learning speed variable
learning_speed_df[['example_mouse']] = 0
learning_speed_df.loc[132,'example_mouse'] = 1
learning_speed_df.loc[132]
learning_speed_df_2 = learning_speed_df[['learning_speed_1','learning_speed_2', 'sessions_to_trained', 'sessions_to_eighty', 'example_mouse']]
learning_speed_df_3 = learning_speed_df[['learning_speed_1','learning_speed_2', 'reverse_sessions_to_trained', 'reverse_sessions_to_eighty', 'example_mouse']]
consistency_df[['example_mouse']] = 0
consistency_df.loc[180,'example_mouse'] = 1
consistency_df_2 = consistency_df[['time_consistency_1', 'time_consistency_2', 'example_mouse']]
consistency_df_3 = consistency_df[['time_consistency_1', 'reverse_time_consistency_2', 'example_mouse']]


# Analyse the different learning speed variables in a pair plot
sns.pairplot(learning_speed_df[['learning_speed_1','learning_speed_2', 'sessions_to_trained', 'sessions_to_eighty']])
plt.show()
        # The higher the speed variables are, the lower the sessions to trained / eighty percent variables should be

# Analyse the different learning speed variables in a pair plot - with example mouse
sns.pairplot(data = learning_speed_df_2, hue = 'example_mouse')
plt.show()


# Analyse the different learning speed variables in a pair plot with reversed sessions_to_trained/ sessions_to_eighty
sns.pairplot(learning_speed_df[['learning_speed_1','learning_speed_2', 'reverse_sessions_to_trained', 'reverse_sessions_to_eighty']])
plt.show()
        # The higher the speed variables are, the lower the sessions to trained / eighty percent variables should be


# Analyse the different learning speed variables in a pair plot with reversed sessions_to_trained/ sessions_to_eighty - with example mouse
sns.pairplot(data = learning_speed_df_2, hue = 'example_mouse', )
plt.show()


# Analyse the different time consistency variables
sns.pairplot(consistency_df[['time_consistency_1', 'time_consistency_2']])
plt.show()
        # The higher the sd, the lower the time consistency should be
        
# Analyse the different time consistency variables - with example mouse
sns.pairplot(data = consistency_df_2, hue = 'example_mouse', )
plt.show()

# Analyse the different time consistency variables with reversed time_consistency_2
sns.pairplot(consistency_df[['time_consistency_1', 'reverse_time_consistency_2']])
plt.show()
        # The higher the sd, the lower the time consistency should be

# Analyse the different time consistency variables with reversed time_consistency_2 - with example mouse
sns.pairplot(data = consistency_df_3, hue = 'example_mouse', )
plt.show()


#%%  

# Create analysis dataframe with all variables 
training_times_df = pd.merge(average_training_time, most_frequent_time, on = 'subject_uuid') # Merge the average training time and the most frequent training time ..
training_times_df = pd.merge(training_times_df, consistency_df, on = 'subject_uuid') # .. with the consistency per subject data frame

analysis_dataframe = pd.merge(learning_speed_df, training_times_df, on = 'subject_uuid') # Merge all variables created together into one data frame
analysis_dataframe = pd.merge(analysis_dataframe, subjects_light_cycle, on = 'subject_uuid')
analysis_dataframe.columns

#%%  


# Create some explorative plots 

# Add example mouse data to the df_sessions
df_sessions[['example_mouse']] = 0
df_sessions.loc[7079:7095,'example_mouse'] = 1
example_mouse_sessions = df_sessions[df_sessions['subject_uuid'] == 'ed8918e4-6a37-4b3d-9b4d-daa14eca0c70']
example_mouse_sessions[['session_time_hour']].value_counts()


# Plot the performance on easy trials per session for the different labs

# General performance
sns.lineplot(data = df_sessions, x = "session_number", y = "performance_easy") # Displays the performance of all mice per training day (hue = "subject_uuid")
sns.lineplot(data = df_sessions, x = "session_number", y = "performance_easy", hue = 'example_mouse') # Displays the performance of all mice per training day (hue = "subject_uuid")


# Performance per lab
plot = sns.lineplot(data = df_sessions, x = "session_number", y = "performance_easy", hue = 'lab_name', style = 'example_mouse')
sns.move_legend(plot, "upper left", bbox_to_anchor=(1, 1))


# Performance per lab
sns.relplot(data = df_sessions , x = "session_number", y = "performance_easy",
            hue = 'lab_name', col = 'lab_name', col_wrap = 3, kind="line")


# Performance per light cycle
sns.lineplot(data = df_sessions, x = "session_number", y = "performance_easy", hue = 'light_cycle')


# Performance of example mouse
sns.lineplot(data = example_mouse_sessions, x = "session_number", y = "performance_easy")


# Create a separate plot for each lab with a highlight on the example mouse
fig = sns.FacetGrid(df_sessions,
                    col= "lab_name", col_wrap = 3, 
                    sharex = True, sharey = True, aspect = 1, hue = "example_mouse", xlim=[-1, 41.5],)
fig.map(sns.lineplot, "session_number",
        "performance_easy", alpha = 0.3)


# Plot one curve for each animal, one panel per lab (code from IBL data)
fig = sns.FacetGrid(df_sessions,
                    col="lab_name", col_wrap=3,
                    sharex=True, sharey=True, aspect=1, hue="subject_uuid", xlim=[-1, 41.5])
fig.map(sns.lineplot, "session_number",
        "performance_easy", color='gray', alpha=0.3)
fig.map(sns.lineplot, "session_number",
        "performance_easy", color='darkblue')
sns.relplot(data = df_sessions , x = "session_number", y = "performance_easy", col = 'lab_name', col_wrap = 3, kind="line")


# Plot the time consistency
consis = consistency_df[['subject_uuid', 'time_consistency_1', 'time_consistency_2']]
df_sessions = pd.merge(df_sessions, consis, on = 'subject_uuid')
df_sessions.columns


# For consistency 1
sns.lineplot(data = df_sessions, x = "session_number", y = "performance_easy", hue = 'time_consistency_1') # Displays the performance of all mice per training day (hue = "subject_uuid")
plt.show()
sns.relplot(data = df_sessions , x = "session_number", y = "performance_easy",
            hue = 'time_consistency_1', col = 'lab_name', col_wrap = 3, kind = "line")

sns.relplot(data = df_sessions , x = "session_number", y = "performance_easy",
            hue = 'time_consistency_1', col = 'light_cycle', col_wrap = 2, kind = "line")


# For consistency 2
sns.lineplot(data = df_sessions, x = "session_number", y = "performance_easy", hue = 'time_consistency_2') # Displays the performance of all mice per training day (hue = "subject_uuid")
plt.show()
sns.relplot(data = df_sessions , x = "session_number", y = "performance_easy",
            hue = 'time_consistency_2', col = 'lab_name', col_wrap = 3, kind = "line")

sns.relplot(data = df_sessions , x = "session_number", y = "performance_easy", style = 'example_mouse', 
            hue = 'time_consistency_2', col = 'light_cycle', col_wrap = 2, kind = "line")



# Create plot of learning_speed per light cycle
sns.scatterplot(data = analysis_dataframe, x = 'lab_name', y = 'learning_speed_1')
sns.scatterplot(data = analysis_dataframe, x = 'lab_name', y = 'learning_speed_2')
sns.scatterplot(data = analysis_dataframe, x = 'lab_name', y = 'session_to_trained')
sns.scatterplot(data = analysis_dataframe, x = 'lab_name', y = 'sessions_to_eighty')

sns.scatterplot(data = analysis_dataframe, x = 'light_cycle', y = 'learning_speed_1')
sns.scatterplot(data = analysis_dataframe, x = 'light_cycle', y = 'learning_speed_2')
sns.scatterplot(data = analysis_dataframe, x = 'light_cycle', y = 'session_to_trained')
sns.scatterplot(data = analysis_dataframe, x = 'light_cycle', y = 'sessions_to_eighty')

sns.scatterplot(data = analysis_dataframe, x = 'lab_name', y = 'learning_speed_1')
sns.scatterplot(data = analysis_dataframe, x = 'light_cycle', y = 'learning_speed_1')

sns.boxplot(data = analysis_dataframe, x = 'lab_name', y = 'learning_speed_1', hue = 'light_cycle_x')
sns.boxplot(data = analysis_dataframe, x = 'light_cycle_x', y = 'learning_speed_1')

analysis_dataframe.columns

#%%  

# STEP 3: ANALYSIS - // 
from statsmodels.formula.api import ols

ols_model = ols(formula = 'learning_speed ~ time_consistency + light_cycle', data = analysis_dataframe)
ols_result = ols_model.fit()
ols_result.summary()


ols_model_all = ols(formula = 'learning_speed ~ performance_end + performance_start + total_number_sessions + total_number_training_days + average_training_time + most_frequent_time + max + sum + time_consistency + light_cycle',  data = analysis_dataframe)
ols_result_all = ols_model.fit()
ols_result_all.summary()




