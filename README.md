# Advanced Information Retrieval Project Group 13

### Introduction

- The idea of this project is to show that the prediction of future Youtube videos of their negativity or positivity due to their previous videos can be done by using Youtube API for creating the dataset, NLP to preprocess the data, Huggingface to enrich the data and a classifier to predict the outcome.

## Project Git Workflow

### Example
- e.g. Feature "implement NLP" -> create a branch with the name "NLPfuctions" -> implement the functions -> create a merge request to the dev branch
-> let another member take a look at the merge request -> if everything is ok accept -> if the project is finished we merge the dev branch into the main branch and publish the work

### Main Branch
- The main branch is only for the release
- Every participant needs to accept the merge request from the dev branch
- Only the dev branch can be merged into the main branch.

### Dev Branch
- The dev branch is used to merge feature branches
- The merge of a feature branch can be done only if another member accepts the merge request

### Feature Branch
- Feature branches are to be created for every new functionality that needs to be implemented
- After finishing the implementation of the feature create a merge request to the dev

## Project

### Data Gathering

- The data is retrieved from youtube by using the youtube API. Due to the circumstance that youtube only allows 100 comments per video we could gather 18300 comments for the evaluation and prediction.

- If you want to get data from another youtube video you need to change the API key and change the channel ID.

```python
api_key = 'Generated API Key'
channel_IDs = ['Channel ID of your channel']
```

### Terminal Call

```
python3 getdata.py
```

### Data Processing

- The youtube comments are preprocessed for future use in the Huggingface model and the classification.

#### Preprocessing
- removing punctuation
- removing URLs, and emails if there are any
- removing emojis
- detection and removal of non-English comments
- spelling correction
- tokenization
- removing stopwords
- lemmatization

### Terminal Call
```
python3 preprocessing.py
```

### Huggingface model

- The Huggingface model is used to enrich the given dataset with 1 for positive comments and 0 for negative comments.

- The training and testing of the Huggingface model were done with a predefined dataset (Twitter comments) and the evaluation was done with the preprocessed Youtube comments.

- As a baseline model we used SGDClassifier from the sci-kit-learn library.

### Terminal Call
```
run jupyter notebook Implementaion_model.ipynb
```

### Naive Bayes Classifier

- To predict future videos of their negativity or positivity by using the enriched comments from the Huggingface model.
- To create a better visualization of the prediction a linear regression model with the same specifications as for the classification was used.

### Terminal Call
```
python3 bayes.py
```
