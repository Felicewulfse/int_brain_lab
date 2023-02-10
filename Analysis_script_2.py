
# Felice Wulfse
# January 2023

# This script is created to analyse the IBL behavioural data
# We investigate whether the light cycle and training time consistency affect learning speed in mice


#%%  

# STEP 1: SET UP AND LOAD DATA
# %reset -f # Remove list

# Load packages
import datajoint as dj
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import statistics 
import matplotlib as mpl
import pingouin as pg
import scipy.stats as stats
import statsmodels.formula.api as smf
from scipy.stats import f_oneway


# Set R home environment to load pymer4
os.environ['R_HOME'] = "/Library/Frameworks/R.framework/Resources"
import pymer4
from pymer4.models import Lmer


# Read in data
df_subjects = pd.read_excel('/Users/felicewulfse/Library/Mobile Documents/com~apple~CloudDocs/Documents/Universiteit/MSc Applied Cognitive Psychology/Thesis/My_repo_int_brain_lab/df_subjects.xlsx')
df_sessions = pd.read_excel('/Users/felicewulfse/Library/Mobile Documents/com~apple~CloudDocs/Documents/Universiteit/MSc Applied Cognitive Psychology/Thesis/My_repo_int_brain_lab/df_sessions.xlsx')

len(df_subjects) # 233 mice
len(df_subjects['subject_uuid'].unique())
len(df_sessions['subject_uuid'].unique()) # 197 mice


#%%  

# STEP 2: CREATE VARIABLES NEEDED FOR THE ANALYSES

# Inspect the different columns in the subjects data frame
df_subjects.columns
df_sessions.columns


# Investigate the subject id's 
len(df_subjects['subject_uuid'].unique()) # 233 unique subject uuid's in the subjects dataframe
len(df_sessions['subject_uuid'].unique()) # 197 unique subject uuid's


# Create the session_date, session_time and session_time hour variable
df_sessions['session_dates'] = pd.to_datetime(df_sessions['session_start_time']).dt.date
df_sessions['session_time'] = pd.to_datetime(df_sessions['session_start_time']).dt.time
df_sessions['session_time_hour'] =  pd.to_datetime(df_sessions['session_start_time']).dt.round('H').dt.hour


# Create lab name data frame to add lab name to sessions dataframe
lab_name = df_subjects[['subject_uuid', 'lab_name']]
df_sessions = pd.merge(df_sessions,lab_name, on = 'subject_uuid')


# Remove two labs based on their 'mixed' light cycle
df_sessions = df_sessions.loc[df_sessions['lab_name'] != 'hoferlab']
df_sessions = df_sessions.loc[df_sessions['lab_name'] != 'mrsicflogellab']


# Create a lab_number and light_cycle variable per lab
labs_df = pd.DataFrame({'lab_number':[1,2,3,4,5,6,7,8],
                                      'light_cycle':[1,1,1,1,0,0,0,0],
                                       'lab_name': ['angelakilab','wittenlab','churchlandlab','churchlandlab_ucla',
                                                    'cortexlab','danlab', 'mainenlab','zadorlab']}) # Merge the lab number and light cycle values to the lab_names
df_subjects = pd.merge(df_subjects, labs_df, on = 'lab_name') # Save the lab number and light cycle values to the subjects data frame
df_sessions = pd.merge(df_sessions, labs_df, on = 'lab_name') # Save the lab number and light cycle values to the sessions data frame


# Create a total_number_sessions and total_number_training_days varaible per subject variable
total_number_sessions = df_sessions.groupby(['subject_uuid'])['session_start_time'].count().reset_index(name = 'total_number_sessions') # Count the total number of sessions for each subject
total_number_training_days = df_sessions.groupby(['subject_uuid'])['session_dates'].count().reset_index(name = 'total_number_training_days') # Count the total number of training days for each subject


# Save the session_number and training_day variable for the session data frame
List_subject_uuids = df_sessions['subject_uuid'].unique() # Create a list with all unique subject_uuid's in the sessions data frame
filters = df_sessions['subject_uuid'].isin(List_subject_uuids) 
df_sessions.loc[filters,'session_number'] = (
    df_sessions.groupby([filters,'subject_uuid'])['session_start_time'].transform(lambda x: pd.CategoricalIndex(x).codes + 1)) # Save the session number for each subject
df_sessions.loc['training_day'] = (
    df_sessions.groupby([filters,'subject_uuid'])['session_dates'].transform(lambda x: pd.CategoricalIndex(x).codes + 1)) # Save the training day for each subject


# Save the subjects and the total number of sessions they had
subjects_session_number = df_sessions[['subject_uuid', 'session_number']]
subjects_session_number = subjects_session_number.groupby(['subject_uuid'], sort=False)['session_number'].max()
               

# Create an average training time per subject variable
average_training_time = df_sessions.groupby('subject_uuid')['session_time_hour'].mean().reset_index(name = 'average_training_time') # Save the average training time for each subject


# Create a most common training time per subject variable
time_frequency = df_sessions.groupby('subject_uuid')['session_time_hour'].value_counts().reset_index(name = 'frequency') # Count the occurences of the session_time_hour
most_frequent_time = time_frequency.groupby('subject_uuid').max().reset_index() # Save the maximum frequency of the session_time_hour
most_frequent_time = most_frequent_time.iloc[:,0:2] # Only save the subject_uuid and session_time_hour
most_frequent_time.rename(columns = {'session_time_hour':'most_frequent_time'}, inplace = True) # Change the column name to most_frequent_time

#%%  


## CREATE TRAINING TIME CONSISTENCY VARIABLES ##

# 1: Consistency training time, defined as the max_time_frequency divided by the sum_time_frequency
max_per_subject = time_frequency.groupby(['subject_uuid'])['frequency'].max().reset_index(name = 'max') # Save the maximum frequency of the training time for each subject
sum_per_subject = time_frequency.groupby(['subject_uuid'])['frequency'].sum().reset_index(name = 'sum') # Save the total frequency of the training time for each subject
consistency_df = pd.merge(max_per_subject, sum_per_subject, on = 'subject_uuid') # Merge the two data frames together
consistency_df['time_consistency_1'] = consistency_df['max']/consistency_df['sum'] # Use the maximum frequency and the total frequency of the training time to calculate the consistency in training time for each subject


# 2: Consistency training time, defined as the standard deviation of the session_time_hour distribution
df_sessions_time = df_sessions[['subject_uuid', 'session_time_hour']] # Create a new data frame with only the subject_uuid and the session_time_hour
sd_time_hour = df_sessions_time.groupby(['subject_uuid'])['session_time_hour'].std().reset_index(name = 'time_consistency_2')
sd_time_hour = pd.DataFrame(sd_time_hour) # Save the values as a dataframe
consistency_df = pd.merge(consistency_df, sd_time_hour, on = 'subject_uuid') # Merge the two data frames together


# Inspect the two different definitions
consistency_df.columns # Check columns
np.corrcoef(consistency_df['time_consistency_1'], consistency_df['time_consistency_2']) # Calculate the correlation
 
#%%  

## CREATE LEARNING SPEED VARIABLES ##

# 1: Learning speed, defined as the max_performance minus the min_performance divided by the total number of sessions
performance_start = df_sessions.groupby(['subject_uuid'])['performance_easy'].min().reset_index(name = 'performance_start') # Save the minimum performance for each subject
performance_end = df_sessions.groupby(['subject_uuid'])['performance_easy'].max().reset_index(name = 'performance_end') # Save the maximum performance for each subject

learning_speed_df = pd.merge(performance_end, performance_start, on = 'subject_uuid') # Merge the data frames with start and end performance for each subject
learning_speed_df = pd.merge(learning_speed_df, total_number_sessions, on = 'subject_uuid') # Also save the total number of sessions to this data frame
learning_speed_df = pd.merge(learning_speed_df, total_number_training_days, on = 'subject_uuid') # Also save the total number of training days to this data frame
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
learning_speed_df = pd.merge(learning_speed_df, performance_sessions, on = 'subject_uuid') # Merge the performance start and end to the learning speed data frame
learning_speed_df['learning_speed_2'] = (learning_speed_df['performance_session_end'] - learning_speed_df['performance_session_start'])/learning_speed_df['total_number_sessions'] # Create the training_speed variable by dividing the performance difference by the total number of sessions for each individual


# 3: Learning speed, defined by the number of sessions untill achieved trained_1a/ trained_1b
trained_animals = df_sessions.loc[(df_sessions['training_status'] == 'trained_1a') | (df_sessions['training_status'] == 'trained_1b')] # Save the rows where the training_status is trained_1a or trained_1b
trained_animals_only = trained_animals.groupby('subject_uuid')['session_number'].min().reset_index(name = 'sessions_to_trained') # Save the minimal session_number per subject 
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

## CREATE ONE DATAFRAME FOR FINAL ANALYSES AND PLOTS ##

# Add the definitions for learning speed and consistency to the subjects data frame
learning_speed_merge = learning_speed_df[['subject_uuid', 'learning_speed_1', 'learning_speed_2', 'sessions_to_trained', 'sessions_to_eighty']]
consistency_merge = consistency_df[['subject_uuid', 'time_consistency_1', 'time_consistency_2']]
mix_merge = pd.merge(learning_speed_merge, consistency_merge, on = 'subject_uuid') # Save the learning speed variables and training time consistency variables
mix_merge_2 = pd.merge(subjects_session_number, mix_merge, on = 'subject_uuid') # Merge with session number per subject
df_subjects_complete = pd.merge(df_subjects, mix_merge_2, on = 'subject_uuid') # Merge with subjects
df_subjects_complete['lab_name'] = df_subjects_complete['lab_name'].str.capitalize() # Save the lab names with capital letter
df_subjects_complete.columns # Check the dataframe


# Remove the subjects that are not in the subjects dataframe, but are in the sessions dataframe
df_subjects_learning_speed_1 = df_subjects_complete[['subject_uuid', 'learning_speed_1']]
df_sessions_complete = pd.merge(df_sessions, df_subjects_learning_speed_1, on = 'subject_uuid') # Now both dataframes contain 124 subjects
len(df_sessions_complete['subject_uuid'].unique()) # Check the number of unique id's
df_sessions_complete[['Percentage correct']] = df_sessions_complete[['performance_easy']] * 100 # Save percentage correct
df_sessions_complete['lab_name'] = df_sessions_complete['lab_name'].str.capitalize() # Save lab names with capital letter
mix_merge_3 = df_subjects_complete[['subject_uuid', 'time_consistency_1', 'time_consistency_2']]
df_sessions_complete = pd.merge(df_sessions_complete, mix_merge_3, on = 'subject_uuid')


# Get some statistics
df_subjects_complete.groupby('lab_name')['subject_uuid'].count() # Get the number of subjects per lab
df_subjects_complete.groupby('lab_name')['session_number'].mean() # Get the average number of sessions per lab
df_subjects_complete['session_number'].mean() # Get the average number of sessions in total
df_subjects_complete['sessions_to_trained'].mean() # Get the average number of sessions in total
df_subjects_complete['sessions_to_trained'].min() # Get the min number of sessions in total
df_subjects_complete['sessions_to_trained'].max() # Get the max number of sessions in total


# Create variable for pltos
lab_order = ["Cortexlab","Danlab","Mainenlab","Zadorlab","Angelakilab","Wittenlab","Churchlandlab"] # Save the lab order for the plots


# Create new variable to divide between low and high consistency (based on medians)
df_subjects_complete['time_consistency_cat_1'] = 'low'
df_subjects_complete['time_consistency_cat_2'] = 'high'
df_subjects_complete.loc[df_subjects_complete['time_consistency_1'] > 0.375, 'time_consistency_cat_1'] = 'High'
df_subjects_complete.loc[df_subjects_complete['time_consistency_2'] > 1.52, 'time_consistency_cat_2'] = 'Low'


# Create new variable to divide between low and high consistency (based on medians)
df_sessions_complete['time_consistency_cat_1'] = 'low'
df_sessions_complete['time_consistency_cat_2'] = 'high'
df_sessions_complete.loc[df_sessions_complete['time_consistency_1'] > 0.375, 'time_consistency_cat_1'] = 'High'
df_sessions_complete.loc[df_sessions_complete['time_consistency_2'] > 1.52, 'time_consistency_cat_2'] = 'Low'


# Create new variable for light cycle
df_sessions_complete["Light cycle"] = df_sessions_complete['light_cycle']
df_sessions_complete.loc[df_sessions_complete['Light cycle'] == 0.0,'Light cycle'] = "NIV"
df_sessions_complete.loc[df_sessions_complete['Light cycle'] == 1.0,'Light cycle'] = "IV"

df_subjects_complete["Light cycle"] = df_subjects_complete['light_cycle']
df_subjects_complete.loc[df_subjects_complete['Light cycle'] == 0.0,'Light cycle'] = "NIV"
df_subjects_complete.loc[df_subjects_complete['Light cycle'] == 1.0,'Light cycle'] = "IV"


#%% 

## CREATE EXAMPLE MICE ##

# Find the indices of the example mice
np.where(df_subjects_complete['subject_uuid'] == '3eee82bf-8ccf-42ac-9e26-472caed5f265') # 5
np.where(df_subjects_complete['subject_uuid'] == '8b79e73e-20f9-4bdc-a78e-341511d893ca') # 76 
np.where(df_subjects_complete['subject_uuid'] == '5f756084-afaf-4c88-96b5-ed29401dbd2f') # 120
np.where(df_subjects_complete['subject_uuid'] == 'dffc24bc-bd97-4c2a-bef3-3e9320dc3dd7') # 15


# Set example mice values to 0-4
df_subjects_complete['example_mouse'] = 0
df_subjects_complete.loc[5,'example_mouse'] = 1
df_subjects_complete.loc[76,'example_mouse'] = 2
df_subjects_complete.loc[120,'example_mouse'] = 3
df_subjects_complete.loc[15,'example_mouse'] = 4


# Set example mice values to 0/1
df_subjects_complete['example_mouse_2'] = 0
df_subjects_complete.loc[5,'example_mouse_2'] = 1
df_subjects_complete.loc[76,'example_mouse_2'] = 1
df_subjects_complete.loc[120,'example_mouse_2'] = 1
df_subjects_complete.loc[15,'example_mouse_2'] = 1


# Find the indices of the example mice (in sessions data frame)
np.where(df_sessions_complete['subject_uuid'] == '3eee82bf-8ccf-42ac-9e26-472caed5f265') # 258:275
np.where(df_sessions_complete['subject_uuid'] == '8b79e73e-20f9-4bdc-a78e-341511d893ca') # 3666:3691
np.where(df_sessions_complete['subject_uuid'] == '5f756084-afaf-4c88-96b5-ed29401dbd2f') # 4365:4392
np.where(df_sessions_complete['subject_uuid'] == 'dffc24bc-bd97-4c2a-bef3-3e9320dc3dd7') # 2740:2842


# Set example mice values to 0-4 
df_sessions_complete[['example_mouse']] = 0
df_sessions_complete.loc[258:275,'example_mouse'] = 1
df_sessions_complete.loc[3261:3286,'example_mouse'] = 2
df_sessions_complete.loc[3896:3923,'example_mouse'] = 3
df_sessions_complete.loc[2390:2492,'example_mouse'] = 4


# Set example mice values to 0/1
df_sessions_complete[['example_mouse_2']] = 0
df_sessions_complete.loc[258:275,'example_mouse_2'] = 1
df_sessions_complete.loc[3261:3286,'example_mouse_2'] = 1
df_sessions_complete.loc[3896:3923,'example_mouse_2'] = 1
df_sessions_complete.loc[2390:2492,'example_mouse_2'] = 1


# Create new variable for example mice in df_sessions
df_sessions_complete["Example mouse"] = df_sessions_complete['example_mouse']
df_sessions_complete.loc[df_sessions_complete['Example mouse'] == 0,'Example mouse'] = "All other mice"
df_sessions_complete.loc[df_sessions_complete['Example mouse'] == 1,'Example mouse'] = "Example mouse 1"
df_sessions_complete.loc[df_sessions_complete['Example mouse'] == 2,'Example mouse'] = "Example mouse 2"
df_sessions_complete.loc[df_sessions_complete['Example mouse'] == 3,'Example mouse'] = "Example mouse 3"
df_sessions_complete.loc[df_sessions_complete['Example mouse'] == 4,'Example mouse'] = "Example mouse 4"

# Create new variable for example mice in df_subjects
df_subjects_complete["Example mouse"] = df_subjects_complete['example_mouse']
df_subjects_complete.loc[df_subjects_complete['Example mouse'] == 0,'Example mouse'] = "All other mice"
df_subjects_complete.loc[df_subjects_complete['Example mouse'] == 1,'Example mouse'] = "Example mouse 1"
df_subjects_complete.loc[df_subjects_complete['Example mouse'] == 2,'Example mouse'] = "Example mouse 2"
df_subjects_complete.loc[df_subjects_complete['Example mouse'] == 3,'Example mouse'] = "Example mouse 3"
df_subjects_complete.loc[df_subjects_complete['Example mouse'] == 4,'Example mouse'] = "Example mouse 4"


# Create dataframes for plots (for learning speed/ time consistency)
df_subjects_complete_2 = df_subjects_complete[['learning_speed_1','learning_speed_2', 'sessions_to_trained', 'sessions_to_eighty', 'Example mouse']]
df_subjects_complete_3 = df_subjects_complete[['time_consistency_1', 'time_consistency_2', 'Example mouse']]


# Get example mice data frame
df_sessions_example_mice = df_sessions_complete[df_sessions_complete['example_mouse_2'] == 1]
example_mice_subjects = df_subjects_complete.loc[df_subjects_complete['example_mouse_2'] == 1]
example_mice_statistics = example_mice_subjects[['subject_uuid', 'example_mouse','learning_speed_1', 'learning_speed_2', 'sessions_to_trained', 'sessions_to_eighty', 'time_consistency_1', 'time_consistency_2']]


example_mice_statistics.groupby(['example_mouse'])['learning_speed_1'].mean()
example_mice_statistics.groupby(['example_mouse'])['learning_speed_2'].mean()
example_mice_statistics.groupby(['example_mouse'])['sessions_to_trained'].mean()
example_mice_statistics.groupby(['example_mouse'])['sessions_to_eighty'].mean()

example_mice_statistics.groupby(['example_mouse'])['time_consistency_1'].mean()
example_mice_statistics.groupby(['example_mouse'])['time_consistency_2'].mean()


#%% 

# STEP 3: CREATE EXPLORATIVE PLOTS 

## METHODS ##

# FIGURE 1: Performance of example mice 2 
sns.set_theme(style = 'darkgrid')
example_mouse_2 = df_sessions_complete.loc[df_sessions_complete['example_mouse'] == 2]
fig_1 = sns.lineplot(data = example_mouse_2, x = "session_number", y = "Percentage correct", palette = 'colorblind')
fig_1.set(xlabel = 'Session number', ylabel = 'Percentage correct', title ='Performance of example mouse 2')
plt.show()


# FIGURE 2: Pairplot different learning speed variables with example mouse
sns.set_theme(style = 'darkgrid')
fig_2 = sns.pairplot(data = df_subjects_complete_2, kind = 'reg', plot_kws = {'line_kws':{'color':'lightskyblue'}}, 
             hue = 'Example mouse', hue_order = ['All other mice', 'Example mouse 1', 'Example mouse 2', 'Example mouse 3', 'Example mouse 4'], diag_kind = 'hist', palette = 'colorblind', corner = True)
sns.move_legend(fig_2, "upper right", bbox_to_anchor=(.82, 0.47))
plt.show(fig_2)


# FIGURE 3: Learning speed boxplot per lab
fig_3a = sns.boxplot(x = 'lab_name', y = 'learning_speed_1', data = df_subjects_complete, palette = 'colorblind', order = lab_order)
plt.xticks(rotation = 90)
fig_3a.set(xlabel = "Lab name")

fig_3b = sns.boxplot(x = 'lab_name', y = 'learning_speed_2', data = df_subjects_complete, palette = 'colorblind', order = lab_order)
plt.xticks(rotation = 90)
fig_3a.set(xlabel = "Lab name")

fig_3c = sns.boxplot(x = 'lab_name', y = 'sessions_to_trained', data = df_subjects_complete, palette = 'colorblind', order = lab_order)
plt.xticks(rotation = 90)
fig_3c.set(xlabel = "Lab name")

fig_3d = sns.boxplot(x = 'lab_name', y = 'sessions_to_eighty', data = df_subjects_complete, palette = 'colorblind', order = lab_order)
plt.xticks(rotation = 90)
fig_3d.set(xlabel = "Lab name")


# FIGURE 4: Pairplot different time consistency variables with example mouse
fig_4 = sns.pairplot(data = df_subjects_complete_3, kind = 'reg', plot_kws = {'line_kws':{'color':'lightskyblue'}}, 
             hue = 'Example mouse', hue_order = ['All other mice', 'Example mouse 1', 'Example mouse 2', 'Example mouse 3', 'Example mouse 4'], diag_kind = 'hist', palette = 'colorblind', corner = True)
sns.move_legend(fig_4, "upper right", bbox_to_anchor=(.7, 0.9))
plt.show(fig_4)



# FIGURE 5: CREATED IN WORD


# FIGURE 6: Normality plots for learning speed
fig_6a = sns.distplot(x = df_subjects_complete['learning_speed_1'])
fig_6a.set(xlabel='learning_speed_1')
fig_6b = sns.distplot(x = df_subjects_complete['learning_speed_2'])
fig_6b.set(xlabel='learning_speed_2')
fig_6c = sns.distplot(x = df_subjects_complete['sessions_to_trained'])
fig_6c.set(xlabel='sessions_to_trained')
fig_6d = sns.distplot(x = df_subjects_complete['sessions_to_eighty'])
fig_6d.set(xlabel='sessions_to_eighty')



## GENERAL PERFORMANCE PLOTS ##

# FIGURE 7: General performance
sns.set_theme(style = 'darkgrid')
fig_7 = sns.lineplot(data = df_sessions_complete, x = 'session_number', y = 'Percentage correct', palette = 'colorblind') 
fig_7.set(xlabel ="Session number", ylabel = "Percentage correct", title ='Performance across all labs and all mice')
plt.show()


# FIGURE 8: Performance per lab
fig_8 = sns.relplot(data = df_sessions_complete , x = 'session_number', y = 'Percentage correct',
            hue = 'lab_name', col = 'lab_name', col_wrap = 4, col_order = lab_order, kind = 'line', palette = 'colorblind', legend = False)
fig_8.set_xlabels("Session number", fontsize = 15)
fig_8.set_ylabels("Percentage correct", fontsize = 15)
for i in range(7):
    fig_8.axes[i].set_title(lab_order[i], fontsize = 20)
plt.show()


fig_8b = sns.relplot(data = df_sessions_complete , x = 'session_number', y = 'Percentage correct',
             hue = 'lab_name', col = 'lab_name', col_wrap = 4, col_order = lab_order, estimator = None, kind = 'line', palette = 'colorblind', legend = False, alpha = 1)
fig_8b.set_xlabels("Session number", fontsize = 15)
fig_8b.set_ylabels("Percentage correct", fontsize = 15)
for i in range(7):
    fig_8b.axes[i].set_title(lab_order[i], fontsize = 20)
plt.show()



# FIGURE 9: Session number distribution of different labs
sns.set_theme(style = 'darkgrid')
fig_9 = sns.displot(
        df_subjects_complete, x = 'session_number', col = 'lab_name', col_wrap = 4, hue = 'lab_name', bins = 1, 
        binwidth = 3, height = 3, facet_kws = dict(margin_titles = True), palette = 'colorblind', col_order = lab_order, legend = False)
fig_9.set_xlabels('Session number')
for i in range(7):
    fig_9.axes[i].set_title(lab_order[i], fontsize = 15)



## EXAMPLE MICE PLOTS ##

# FIGURE 10: Performance of example mice (without general performance)
sns.set_theme(style = 'darkgrid')
fig_10 = sns.lineplot(data = df_sessions_example_mice, x = 'session_number', y = 'Percentage correct', hue = 'Example mouse', hue_order = ['Example mouse 1', 'Example mouse 2', 'Example mouse 3', 'Example mouse 4'],palette = 'colorblind')
fig_10.set(xlabel = 'Session number', ylabel = 'Percentage correct', title = 'Performance of the example mice')
plt.show()



# FIGURE 11: Training time distribution of example mice
sns.set_theme(style = 'darkgrid')
fig_11 = sns.displot(
    df_sessions_example_mice, x = 'session_time_hour', col = 'example_mouse', col_wrap = 2,hue = 'Light cycle', hue_order = ['NIV','IV'], bins = 1,
    binwidth = 1, height = 3, facet_kws = dict(margin_titles = True), palette = 'colorblind')
fig_11.set_xlabels('Session start time (hours)', fontsize = 10)
example_mice_order = ['Example mouse 1', 'Example mouse 2', 'Example mouse 3', 'Example mouse 4']
for i in range(4):
    fig_11.axes[i].set_title(example_mice_order[i])




## LIGHT CYCLE PLOTS ##

# FIGURE 12: Performance per lab and light_cycle 
fig_12 = sns.relplot(data = df_sessions_complete, x = 'session_number', y = 'Percentage correct', 
            hue = 'Light cycle', hue_order = ['NIV','IV'], 
            col = 'lab_name', col_wrap = 4, col_order = lab_order, 
            kind = 'line', palette = 'colorblind')
fig_12.set_xlabels('Session number', fontsize = 15)
fig_12.set_ylabels('Percentage correct', fontsize = 15)
for i in range(7):
    fig_12.axes[i].set_title(lab_order[i], fontsize = 20)
sns.move_legend(fig_12, "lower right", bbox_to_anchor = (0.8, 0.35))


# FIGURE 13: Performance per light cycle
fig_13 = sns.lineplot(data = df_sessions_complete, x = 'session_number', y = 'Percentage correct', hue = 'Light cycle', hue_order = ['NIV', 'IV'], palette = 'colorblind')
fig_13.set(xlabel = 'Session number', ylabel = 'Percentage correct', title = "Performance for labs with NIV/ IV light cycle")



# FIGURE 14: Learning speed per light cycle 
sns.boxplot(x = 'Light cycle', y = 'learning_speed_1', data = df_subjects_complete, palette = 'pastel')
fig_14a = sns.stripplot(x ='Light cycle', y = 'learning_speed_1', data = df_subjects_complete, hue = 'lab_name')
fig_14a.set(xlabel = 'Light cycle', ylabel = 'LS1')
plt.legend(title = 'Lab name')
sns.move_legend(fig_14a, "upper left", bbox_to_anchor = (1, 1))


sns.boxplot(x ='Light cycle', y = 'learning_speed_2', data = df_subjects_complete, palette = 'pastel')
fig_14b = sns.stripplot(x = 'Light cycle', y = 'learning_speed_2', data = df_subjects_complete, hue = 'lab_name')
fig_14b.set(xlabel = 'Light cycle', ylabel = 'LS2')
plt.legend(title = 'Lab name')
sns.move_legend(fig_14b, "upper left", bbox_to_anchor = (1, 1))


sns.boxplot(x ='Light cycle', y = 'sessions_to_trained', data = df_subjects_complete, palette = 'pastel')
fig_14c = sns.stripplot(x = 'Light cycle', y = 'sessions_to_trained', data = df_subjects_complete, hue = 'lab_name')
fig_14c.set(xlabel = 'Light cycle', ylabel = 'LS3')
plt.legend(title = 'Lab name')
sns.move_legend(fig_14c, "upper left", bbox_to_anchor = (1, 1))


sns.boxplot(x ='Light cycle', y = 'sessions_to_eighty', data = df_subjects_complete, palette = 'pastel')
fig_14d = sns.stripplot(x = 'Light cycle', y = 'sessions_to_eighty', data = df_subjects_complete, hue = 'lab_name')
fig_14d.set(xlabel = 'Light cycle', ylabel = 'LS4')
plt.legend(title = 'Lab name')
sns.move_legend(fig_14d, "upper left", bbox_to_anchor = (1, 1))



# FIGURE 15: Learning speed boxplot per lab and light cycle
sns.boxplot(x = 'lab_name', y = 'learning_speed_1', data = df_subjects_complete, palette = 'pastel', hue = 'Light cycle', order = lab_order)
fig_15a = sns.stripplot(x = 'lab_name', y = 'learning_speed_1', data = df_subjects_complete, order = lab_order)
sns.move_legend(fig_15a, "upper left", bbox_to_anchor=(1, 1))
plt.xticks(rotation = 90)
fig_15a.set(xlabel = 'Lab name', ylabel  = 'LS1')


sns.boxplot(x = 'lab_name', y = 'learning_speed_2', data = df_subjects_complete, palette = 'pastel', hue = 'Light cycle',order = lab_order)
fig_15b = sns.stripplot(x = 'lab_name', y = 'learning_speed_2', data = df_subjects_complete, order = lab_order)
sns.move_legend(fig_15b, "upper left", bbox_to_anchor=(1, 1))
plt.xticks(rotation = 90)
fig_15b.set(xlabel = 'Lab name', ylabel  = 'LS2')


sns.boxplot(x = 'lab_name', y = 'sessions_to_trained', data = df_subjects_complete, palette = "pastel", hue = 'Light cycle', order = lab_order)
fig_15c = sns.stripplot(x = 'lab_name', y = 'sessions_to_trained', data = df_subjects_complete, order = lab_order)
sns.move_legend(fig_15c, "upper left", bbox_to_anchor=(1, 1))
plt.xticks(rotation = 90)
fig_15c.set(xlabel = 'Lab name', ylabel  = 'LS3')


sns.boxplot(x = 'lab_name', y = 'sessions_to_eighty', data = df_subjects_complete, palette = "pastel",hue = 'Light cycle', order = lab_order)
fig_15d = sns.stripplot(x = 'lab_name', y = 'sessions_to_eighty', data = df_subjects_complete, order = lab_order)
sns.move_legend(fig_15d, "upper left", bbox_to_anchor=(1, 1))
plt.xticks(rotation = 90)
fig_15d.set(xlabel = 'Lab name', ylabel  = 'LS4')



## TRAINING TIME CONSISTENCY PLOTS ##

# FIGURE 16: Time distribution of all labs
sns.set_theme(style = 'darkgrid')
fig_16 = sns.displot(
    df_sessions_complete, x = 'session_time_hour', col = 'lab_name', col_wrap = 4, hue = 'Light cycle', col_order = lab_order,
    binwidth = 1, height = 3,facet_kws = dict(margin_titles = True), palette = 'colorblind')
fig_16.set_xlabels('Session start time (hours)', fontsize = 10)
for i in range(7):
    fig_16.axes[i].set_title(lab_order[i], fontsize = 15) 
sns.move_legend(fig_16, "lower right", bbox_to_anchor = (0.8, 0.35))


# FIGURE 17: Time consistency per lab
sns.boxplot(x ='lab_name', y = 'time_consistency_1', data = df_subjects_complete, palette = 'pastel')
fig_17a = sns.stripplot(x ='lab_name', y = 'time_consistency_1', data = df_subjects_complete, hue = 'lab_name')
plt.xticks(rotation = 90)
fig_17a.set(xlabel = 'Lab name', ylabel = 'TC1')
sns.move_legend(fig_17a, "upper left", bbox_to_anchor=(1, 1))


sns.boxplot(x ='lab_name', y = 'time_consistency_2', data = df_subjects_complete, palette = 'pastel')
fig_17b = sns.stripplot(x ='lab_name', y = 'time_consistency_2', data = df_subjects_complete, hue = 'lab_name')
plt.xticks(rotation = 90)
fig_17b.set(xlabel = 'Lab name', ylabel = 'TC2')
sns.move_legend(fig_17b, "upper left", bbox_to_anchor=(1, 1))



# FIGURE 18: Performance and categorical time consistency 1 and 2
fig_18a = sns.lineplot(data = df_sessions_complete, x = "session_number", y = "Percentage correct", 
             hue = 'time_consistency_cat_1', palette = 'colorblind') # Displays the performance of all mice per training day (hue = "subject_uuid")
fig_18a.set(xlabel = 'Session number')
plt.legend(title = 'Training time consistency')
sns.move_legend(fig_18a, "upper left", bbox_to_anchor=(1, 1))


fig_18b = sns.lineplot(data = df_sessions_complete, x = "session_number", y = "Percentage correct", hue = 'time_consistency_cat_2', palette = "colorblind") # Displays the performance of all mice per training day (hue = "subject_uuid")
fig_18b.set(xlabel = 'Session number')
plt.legend(title = 'Training time consistency')
sns.move_legend(fig_18b, "upper left", bbox_to_anchor=(1, 1))



# FIGURE 19: Time consistency boxplots per lab
sns.boxplot(x ='light_cycle', y = 'time_consistency_1', data = df_subjects_complete, palette = 'colorblind')
sns.boxplot(x ='light_cycle', y = 'time_consistency_2', data = df_subjects_complete, palette = 'colorblind')



# FIGURE 20: Effect of time consistency on learning speed by light cycle
fig_20a = sns.scatterplot(x ='time_consistency_1', y = 'learning_speed_1', data = df_subjects_complete, palette = 'colorblind', hue = 'Light cycle')
fig_20a.set(xlabel = 'TC1', ylabel = "LS1")

fig_20b = sns.scatterplot(x ='time_consistency_1', y = 'learning_speed_2', data = df_subjects_complete, palette = 'colorblind', hue = 'Light cycle')
fig_20b.set(xlabel = 'TC1', ylabel = "LS2")

fig_20c = sns.scatterplot(x ='time_consistency_1', y = 'sessions_to_trained', data = df_subjects_complete, palette = 'colorblind', hue = 'Light cycle')
fig_20c.set(xlabel = 'TC1', ylabel = "LS3")

fig_20d = sns.scatterplot(x ='time_consistency_1', y = 'sessions_to_eighty', data = df_subjects_complete, palette = 'colorblind', hue = 'Light cycle')
fig_20d.set(xlabel = 'TC1', ylabel = "LS4")


fig_20e = sns.scatterplot(x ='time_consistency_2', y = 'learning_speed_1', data = df_subjects_complete, palette = 'colorblind', hue = 'Light cycle')
fig_20e.set(xlabel = 'TC2', ylabel = "LS1")

fig_20f = sns.scatterplot(x ='time_consistency_2', y = 'learning_speed_2', data = df_subjects_complete, palette = 'colorblind', hue = 'Light cycle')
fig_20f.set(xlabel = 'TC2', ylabel = "LS2")

fig_20g = sns.scatterplot(x ='time_consistency_2', y = 'sessions_to_trained', data = df_subjects_complete, palette = 'colorblind', hue = 'Light cycle')
fig_20g.set(xlabel = 'TC2', ylabel = "LS3")

fig_20h = sns.scatterplot(x ='time_consistency_2', y = 'sessions_to_eighty', data = df_subjects_complete, palette = 'colorblind', hue = 'Light cycle')
fig_20h.set(xlabel = 'TC2', ylabel = "LS4")





#%%  

# STEP 4: ANALYSIS


# Create data frames for Kruskal Wallis tests
df_Ange_1 = df_subjects_complete.loc[df_subjects_complete['lab_name'] == 'Angelakilab']['learning_speed_1']
df_Witt_1 = df_subjects_complete.loc[df_subjects_complete['lab_name'] == 'Wittenlab']['learning_speed_1']
df_Chur_1 = df_subjects_complete.loc[df_subjects_complete['lab_name'] == 'Churchlandlab']['learning_speed_1']
df_Cort_1 = df_subjects_complete.loc[df_subjects_complete['lab_name'] == 'Cortexlab']['learning_speed_1']
df_Danl_1 = df_subjects_complete.loc[df_subjects_complete['lab_name'] == 'Danlab']['learning_speed_1']
df_Main_1 = df_subjects_complete.loc[df_subjects_complete['lab_name'] == 'Mainenlab']['learning_speed_1']
df_Zado_1 = df_subjects_complete.loc[df_subjects_complete['lab_name'] == 'Zadorlab']['learning_speed_1']


df_Ange_2 = df_subjects_complete.loc[df_subjects_complete['lab_name'] == 'Angelakilab']['learning_speed_2']
df_Witt_2 = df_subjects_complete.loc[df_subjects_complete['lab_name'] == 'Wittenlab']['learning_speed_2']
df_Chur_2 = df_subjects_complete.loc[df_subjects_complete['lab_name'] == 'Churchlandlab']['learning_speed_2']
df_Cort_2 = df_subjects_complete.loc[df_subjects_complete['lab_name'] == 'Cortexlab']['learning_speed_2']
df_Danl_2 = df_subjects_complete.loc[df_subjects_complete['lab_name'] == 'Danlab']['learning_speed_2']
df_Main_2 = df_subjects_complete.loc[df_subjects_complete['lab_name'] == 'Mainenlab']['learning_speed_2']
df_Zado_2 = df_subjects_complete.loc[df_subjects_complete['lab_name'] == 'Zadorlab']['learning_speed_2']


df_Ange_3 = df_subjects_complete.loc[df_subjects_complete['lab_name'] == 'Angelakilab']['sessions_to_trained']
df_Witt_3 = df_subjects_complete.loc[df_subjects_complete['lab_name'] == 'Wittenlab']['sessions_to_trained']
df_Chur_3 = df_subjects_complete.loc[df_subjects_complete['lab_name'] == 'Churchlandlab']['sessions_to_trained']
df_Cort_3 = df_subjects_complete.loc[df_subjects_complete['lab_name'] == 'Cortexlab']['sessions_to_trained']
df_Danl_3 = df_subjects_complete.loc[df_subjects_complete['lab_name'] == 'Danlab']['sessions_to_trained']
df_Main_3 = df_subjects_complete.loc[df_subjects_complete['lab_name'] == 'Mainenlab']['sessions_to_trained']
df_Zado_3 = df_subjects_complete.loc[df_subjects_complete['lab_name'] == 'Zadorlab']['sessions_to_trained']

df_Ange_4 = df_subjects_complete.loc[df_subjects_complete['lab_name'] == 'Angelakilab']['sessions_to_eighty']
df_Witt_4 = df_subjects_complete.loc[df_subjects_complete['lab_name'] == 'Wittenlab']['sessions_to_eighty']
df_Chur_4 = df_subjects_complete.loc[df_subjects_complete['lab_name'] == 'Churchlandlab']['sessions_to_eighty']
df_Cort_4 = df_subjects_complete.loc[df_subjects_complete['lab_name'] == 'Cortexlab']['sessions_to_eighty']
df_Danl_4 = df_subjects_complete.loc[df_subjects_complete['lab_name'] == 'Danlab']['sessions_to_eighty']
df_Main_4 = df_subjects_complete.loc[df_subjects_complete['lab_name'] == 'Mainenlab']['sessions_to_eighty']
df_Zado_4 = df_subjects_complete.loc[df_subjects_complete['lab_name'] == 'Zadorlab']['sessions_to_eighty']


k_labs_1 = stats.kruskal(df_Ange_1, df_Witt_1, df_Chur_1, df_Cort_1, df_Danl_1, df_Main_1, df_Zado_1)
k_labs_2 = stats.kruskal(df_Ange_2, df_Witt_2, df_Chur_2, df_Cort_2, df_Danl_2, df_Main_2, df_Zado_2)
k_labs_3 = stats.kruskal(df_Ange_3, df_Witt_3, df_Chur_3, df_Cort_3, df_Danl_3, df_Main_3, df_Zado_3)
k_labs_4 = stats.kruskal(df_Ange_4, df_Witt_4, df_Chur_4, df_Cort_4, df_Danl_4, df_Main_4, df_Zado_4)



# Testing the effect of light cycle on learning pseed
# Basic formula: learning_speed ~ 1 + light_cycle + 1|lab_number

# Perform t-test
t_test_a = stats.ttest_ind(df_subjects_complete['learning_speed_1'][df_subjects_complete['light_cycle'] == 0],
                           df_subjects_complete['learning_speed_1'][df_subjects_complete['light_cycle'] == 1])

t_test_b = stats.ttest_ind(df_subjects_complete['learning_speed_2'][df_subjects_complete['light_cycle'] == 0],
                           df_subjects_complete['learning_speed_2'][df_subjects_complete['light_cycle'] == 1])

t_test_c = stats.ttest_ind(df_subjects_complete['sessions_to_trained'][df_subjects_complete['light_cycle'] == 0],
                           df_subjects_complete['sessions_to_trained'][df_subjects_complete['light_cycle'] == 1])

t_test_d = stats.ttest_ind(df_subjects_complete['sessions_to_eighty'][df_subjects_complete['light_cycle'] == 0],
                           df_subjects_complete['sessions_to_eighty'][df_subjects_complete['light_cycle'] == 1])



# Get a mean of the learning speed variables for light cycle 1 and 2 
df_subjects_complete.groupby(['light_cycle'])['learning_speed_1'].mean()
df_subjects_complete.groupby(['light_cycle'])['learning_speed_2'].mean()
df_subjects_complete.groupby(['light_cycle'])['sessions_to_trained'].mean()
df_subjects_complete.groupby(['light_cycle'])['sessions_to_eighty'].mean()


# Get a standard deviation of the learning speed variables for light cycle 1 and 2 
df_subjects_complete.groupby(['light_cycle'])['learning_speed_1'].std()
df_subjects_complete.groupby(['light_cycle'])['learning_speed_2'].std()
df_subjects_complete.groupby(['light_cycle'])['sessions_to_trained'].std()
df_subjects_complete.groupby(['light_cycle'])['sessions_to_eighty'].std()



# Create mixture models
md_lc_1a = Lmer('learning_speed_1 ~ light_cycle + (1|lab_number)', data = df_subjects_complete)
mdf_lc_1a = md_lc_1a.fit()
print(md_lc_1a.coefs)
print(md_lc_1a.fixef)

md_lc_1b = Lmer('learning_speed_2 ~ light_cycle + (1|lab_name)', data = df_subjects_complete)
mdf_lc_1b = md_lc_1b.fit()
print(md_lc_1b.coefs)

md_lc_1c = Lmer('sessions_to_trained ~ light_cycle + (1|lab_name)', data = df_subjects_complete)
mdf_lc_1c = md_lc_1c.fit()
print(md_lc_1c.coefs)

md_lc_1d = Lmer('sessions_to_eighty ~ light_cycle + (1|lab_name)', data = df_subjects_complete)
mdf_lc_1d = md_lc_1d.fit()
print(md_lc_1d.coefs)




# Testing the effect of training time consistency
# Basic formula: learning_speed ~ 1 + time_consistency + 1|lab_number

df_subjects_complete['time_consistency_1'].median()
df_subjects_complete['time_consistency_1'].mean()
df_subjects_complete['time_consistency_2'].median()
df_subjects_complete['time_consistency_2'].mean()


# Get means for low vs high consistency
df_subjects_complete.groupby(['time_consistency_cat_1'])['learning_speed_1'].mean()
df_subjects_complete.groupby(['time_consistency_cat_1'])['learning_speed_2'].mean()
df_subjects_complete.groupby(['time_consistency_cat_1'])['sessions_to_trained'].mean()
df_subjects_complete.groupby(['time_consistency_cat_1'])['sessions_to_eighty'].mean()

df_subjects_complete.groupby(['time_consistency_cat_1'])['learning_speed_1'].std()
df_subjects_complete.groupby(['time_consistency_cat_1'])['learning_speed_2'].std()
df_subjects_complete.groupby(['time_consistency_cat_1'])['sessions_to_trained'].std()
df_subjects_complete.groupby(['time_consistency_cat_1'])['sessions_to_eighty'].std()


df_subjects_complete.groupby(['time_consistency_cat_2'])['learning_speed_1'].mean()
df_subjects_complete.groupby(['time_consistency_cat_2'])['learning_speed_2'].mean()
df_subjects_complete.groupby(['time_consistency_cat_2'])['sessions_to_trained'].mean()
df_subjects_complete.groupby(['time_consistency_cat_2'])['sessions_to_eighty'].mean()


df_subjects_complete.groupby(['time_consistency_cat_2'])['learning_speed_1'].std()
df_subjects_complete.groupby(['time_consistency_cat_2'])['learning_speed_2'].std()
df_subjects_complete.groupby(['time_consistency_cat_2'])['sessions_to_trained'].std()
df_subjects_complete.groupby(['time_consistency_cat_2'])['sessions_to_eighty'].std()


# Perform t-test
t_test_tc_1a = stats.ttest_ind(df_subjects_complete['learning_speed_1'][df_subjects_complete['time_consistency_1'] <= 0.375],
                               df_subjects_complete['learning_speed_1'][df_subjects_complete['time_consistency_1'] > 0.375])

t_test_tc_1b = stats.ttest_ind(df_subjects_complete['learning_speed_2'][df_subjects_complete['time_consistency_1'] <= 0.375],
                               df_subjects_complete['learning_speed_2'][df_subjects_complete['time_consistency_1'] > 0.375])

t_test_tc_1c = stats.ttest_ind(df_subjects_complete['sessions_to_trained'][df_subjects_complete['time_consistency_1'] <= 0.375],
                               df_subjects_complete['sessions_to_trained'][df_subjects_complete['time_consistency_1'] > 0.375])

t_test_tc_1d = stats.ttest_ind(df_subjects_complete['sessions_to_eighty'][df_subjects_complete['time_consistency_1'] <= 0.375],
                               df_subjects_complete['sessions_to_eighty'][df_subjects_complete['time_consistency_1'] > 0.375])


# Perform t-test
t_test_tc_2a = stats.ttest_ind(df_subjects_complete['learning_speed_1'][df_subjects_complete['time_consistency_2'] <= 1.52],
                               df_subjects_complete['learning_speed_1'][df_subjects_complete['time_consistency_2'] > 1.52])

t_test_tc_2b = stats.ttest_ind(df_subjects_complete['learning_speed_2'][df_subjects_complete['time_consistency_2'] <= 1.52],
                               df_subjects_complete['learning_speed_2'][df_subjects_complete['time_consistency_2'] > 1.52])

t_test_tc_2c = stats.ttest_ind(df_subjects_complete['sessions_to_trained'][df_subjects_complete['time_consistency_2'] <= 1.52],
                               df_subjects_complete['sessions_to_trained'][df_subjects_complete['time_consistency_2'] > 1.52])

t_test_tc_2d = stats.ttest_ind(df_subjects_complete['sessions_to_eighty'][df_subjects_complete['time_consistency_2'] <= 1.52],
                               df_subjects_complete['sessions_to_eighty'][df_subjects_complete['time_consistency_2'] > 1.52])



# Now create mixture models
md_tc1_1a = Lmer('learning_speed_1 ~ time_consistency_1 + (1|lab_name)', data = df_subjects_complete)
mdf_tc1_1a = md_tc1_1a.fit()
print(md_tc1_1a.coefs)
print(md_tc1_1a.fixef)

md_tc1_1b = Lmer('learning_speed_2 ~ time_consistency_1 + (1|lab_name)', data = df_subjects_complete)
mdf_tc1_1b = md_tc1_1b.fit()
print(md_tc1_1b.coefs)
print(md_tc1_1b.fixef)

md_tc1_1c = Lmer('sessions_to_trained ~ time_consistency_1 + (1|lab_name)', data = df_subjects_complete)
mdf_tc1_1c = md_tc1_1c.fit()
print(md_tc1_1c.coefs)
print(md_tc1_1c.fixef)

md_tc1_1d = Lmer('sessions_to_eighty ~ time_consistency_1 + (1|lab_name)', data = df_subjects_complete)
mdf_tc1_1d = md_tc1_1d.fit()
print(md_tc1_1d.coefs)
print(md_tc1_1d.fixef)



md_tc2_1a = Lmer('learning_speed_1 ~ time_consistency_2 + (1|lab_name)', data = df_subjects_complete)
mdf_tc2_1a = md_tc2_1a.fit()
print(md_tc2_1a.coefs)
print(md_tc2_1a.fixef)

md_tc2_1b = Lmer('learning_speed_2 ~ time_consistency_2 + (1|lab_name)', data = df_subjects_complete)
mdf_tc2_1b = md_tc2_1b.fit()
print(md_tc2_1b.coefs)
print(md_tc2_1b.fixef)

md_tc2_1c = Lmer('sessions_to_trained ~ time_consistency_2 + (1|lab_name)', data = df_subjects_complete)
mdf_tc2_1c = md_tc2_1c.fit()
print(md_tc2_1c.coefs)
print(md_tc2_1c.fixef)

md_tc2_1d = Lmer('sessions_to_eighty ~ time_consistency_2 + (1|lab_name)', data = df_subjects_complete)
mdf_tc2_1d = md_tc2_1d.fit()
print(md_tc2_1d.coefs)
print(md_tc2_1d.fixef)




# Testing the interaction between light cycles and training time consistency
# Basic formula: learning_speed ~ 1 + light_cycle + time_consistency + 1|lab_number

md_2_1a = Lmer('learning_speed_1 ~ light_cycle + time_consistency_1 + (light_cycle * time_consistency_1) + (1|lab_number)', data = df_subjects_complete)
mdf_2_1a = md_2_1a.fit()
print(md_2_1a.coefs)
print(md_2_1a.fixef)

md_2_1b = Lmer('learning_speed_2 ~ light_cycle + time_consistency_1 + (light_cycle * time_consistency_1) + (1|lab_name)', data = df_subjects_complete)
mdf_2_1b = md_2_1b.fit()
print(md_2_1b.coefs)
print(md_2_1b.fixef)

md_2_1c = Lmer('sessions_to_trained ~ light_cycle + time_consistency_1 + (light_cycle * time_consistency_1) + (1|lab_name)', data = df_subjects_complete)
mdf_2_1c = md_2_1c.fit()
print(md_2_1c.coefs)
print(md_2_1c.fixef)

md_2_1d = Lmer('sessions_to_eighty ~ light_cycle + time_consistency_1 + (light_cycle * time_consistency_1) + (1|lab_name)', data = df_subjects_complete)
mdf_2_1d = md_2_1d.fit()
print(md_2_1d.coefs)
print(md_2_1d.fixef)


md_2_2a = Lmer('learning_speed_1 ~ light_cycle + time_consistency_2 + (light_cycle * time_consistency_2) + (1|lab_name)', data = df_subjects_complete)
mdf_2_2a = md_2_2a.fit()
print(md_2_2a.coefs)
print(md_2_2a.fixef)

md_2_2b = Lmer('learning_speed_2 ~ light_cycle + time_consistency_2 + (light_cycle * time_consistency_2) + (1|lab_name)', data = df_subjects_complete)
mdf_2_2b = md_2_2b.fit()
print(md_2_2b.coefs)
print(md_2_2b.fixef)

md_2_2c = Lmer('sessions_to_trained ~ light_cycle + time_consistency_2 + (light_cycle * time_consistency_2) + (1|lab_name)', data = df_subjects_complete)
mdf_2_2c = md_2_2c.fit()
print(md_2_2c.coefs)
print(md_2_2c.fixef)

md_2_2d = Lmer('sessions_to_eighty ~ light_cycle + time_consistency_2 + (light_cycle * time_consistency_2) + (1|lab_name)', data = df_subjects_complete)
mdf_2_2d = md_2_2d.fit()
print(md_2_2d.coefs)
print(md_2_2d.fixef)




