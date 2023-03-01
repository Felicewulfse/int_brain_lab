## Thesis project Felice Wulfse, March 2023

For my thesis I studied the differences in learning speed across laboratories and between individual mice in the decision-making task of the International Brain Laboratory et al. (2021). More specifically, I investigated the role of the light cycle used during the experiments and training time consistency of the mice on learning speed. My supervisor Dr. A.E. Urai (Cognitive Neuroscience, Leiden University) has guided me through the project. 


The International Brain Laboratory. (2021). Standardized and reproducible measurement of 	decision-making inmice. eLife, (10), 1-28. https://doi.org/10.7554/eLife.84310. 


My contact information: felicewulfse@gmail.com

For more information about the International Brain Laboratory: a.e.urai@fsw.leidenuniv.nl



# Explaining differences in learning speed in the decision-making task of the International Brain laboratory by light cycle and training time consistency

The script _Download_data_ was used to download the data. It follows the steps from https://anne-urai.github.io/lab_wiki/IBLdata_behav.html. I have included it for transparancy. 

The script _Analysis_script_ is used to analyse the data. The script is divided in four parts:
1. Step 1: Set up 
2. Step 2: Data manipulation
3. Step 3: Data visualisation
4. Step 4: Analyses


Step 1: In the first step, I load required packages and I set my environment. Moreover, I read in the data, which I downloaded via https://anne-urai.github.io/lab_wiki/IBLdata_behav.html. After downloading the data, I saved the data frames in excel files, so that I did not have to make a connection to the database each time I would work on the project. My code to download the data can be found in Download_data. 

Step 2: In the second step, I manipulated the data and created variables used for the analyses. 

Step 3: As third, I created the plots used to explain the data.

Step 4: At last, I performed data analysis using Kruskall-Wallis tests, t-tests and mixture analyses. 


# Variables used for the analysis

Light cycle (1 categorical variable): 
- Non-inverted light cycle / Inverted light cycle 

Learning speed (four continuous variables):
- Learning speed 1 =  (performance_last-performance_first)/(num_sessions)
- Learning speed 2 =  (performance_max-performance_min)/(num_sessions)
- Sessions to trained = num_sessions until the subject is trained
- Sessions to eighty = num_sessions until 80% of performance is reached


Training time consistency (two continuous variables):
- Time consistency 1:  (occurences of training on preferred time)/(num_sessions)
- Time consistency 2: standard deviation of time distributions



# The Decision making task of the International Brain Laboratory et al. (2021)
All mice underwent surgery in which a head bar was attached for head-fixation. After surgery, the subjects had a recovery phase. When recovery was over, the mice were set on water control (Urai et al., 2021) and they got habituated to the experiment environment. They started learning the basic task, in which they had to move a wheel to shift the stimulus (a gabor patch presented on either the left or the right side of the screen) to the centre. In the basic task the probability of the stimulus appearing on the left or on the right was equal (0.5/0.5). When they were trained on the basic task, they learned the full task in which the probability of the stimulus appearing on the left or the right switched between 0.2/0.8 and 0.8/0.2. In both the basic and the full task, the contrast changes ranging from -100% to 100% over the trials. Less contrast makes the task more difficult, as it is less visible. Session duration was not fixed but determined by the experimenter. It depended on the performance, but was always stopped after 90 minutes. There were large differences in the number of trials per sessions. Therefore, some mice could have had more trials per session and learn the basic task in fewer sessions (The International Brain Laboratory et al., 2021).


