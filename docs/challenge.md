# Documentation

## 1. Model Selection

#### **With this, the model to be productive must be the one that is trained with the top 10 features and class balancing, but which one?**
#### I would push it back to the DS and request further analysis with the following observations:
- #### taking the top 10 most important features is "dangerous" in this context, because there are airline-related features in the top 10, and the dataset is highly umbalanced airline-wise. Take Grupo LATAM, for example: its flights dominate the flight distribution per airline, so removing several variables and using only 10 would increase the weight of LATAM operation label in the prediction, causing the model to overfit to such airline. A suggestion to the DS would be to curate the most important features to be agnostic to highly umbalanced variables, such as airline. A good pick would be to start with month-related and flight type features (MES_x, TIPOVUELO), since the flight distribution for those is more uniform.
- #### I don't exactly agree with the statement that balancing classes improves the model performance just because it increases the recall of class 1. I'd say it decreases overall model performance (lower f1-score), but on the other hand makes it more robust and less overfit, thus in my opinion making it better, even though the performance is poorer. 

- #### The variable 'training_data' was created, but only the variable 'features' was used to train the model, leaving fields SIGLADES and DIANOM behind, which could be valuable. Why is that?

### Final thoughts
#### I'd suggest keeping balancing/oversampling, choosing more carefully the inputs used and experiment using SIGLADES and DIANOM. Even though I wouldn't use any of the models, for the sake of this challenge, let's proceed with model 4bii (logistic regression with no balancing nor feature selection).

## 2. model.py

#### Thinking of typical use-cases, in model.py were added:
- #### the argument 'export_flag' in method 'fit' to give the user the option of exporting the train and test sets used when fitting the model;
- #### the argument 'persist_flag' in method 'fit' that persists the model by dumping it into a file named as the string used in the argument;
- #### the method 'load' which takes a model file name as argument to directly load the model previously dumped by the use of 'model_name' in method 'fit'. 

## 3. api
#### The api was deployed and passed in all tests.

## 4. CI/CD
#### A cicd workflow was implemented within github actions and sucessfully tests the code and deploys the latest model/API.
