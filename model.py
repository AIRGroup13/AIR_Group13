import torch
import pandas as pd
import pyarrow as pa
import datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from tqdm.auto import tqdm

#for baseline model
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

####################################################################################################################
#Get preprocessed comments
df = pd.read_csv("Comments_prep.csv")
eval_dataset = []
for row in df.iterrows():
    one_video = []
    for comment in row:
      if(type(comment) == int):
        continue
      temp_list = [item for item in comment if not(pd.isnull(item)) == True] #remove NaN comments
      for entry in temp_list:
        if type(entry) is not str: 
          temp_list.remove(entry) #remove column numbers from dataFrame
    eval_dataset.append(temp_list)
    
#print(type(eval_dataset), len(eval_dataset))  


####################################################################################################################
#Distilbert
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
 
####################################################################################################################
#Use GPU for faster training
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
print(device)

###############################################################################################################
#Calculating output before fine-tuning model

outputs_before = []

with torch.no_grad():
    for video in eval_dataset:
      output_one_video = []
      for comment in video:
        input = tokenizer(comment, padding=True, truncation=True, return_tensors='pt')
        input.to(device)
        output = model(**input)
        #Get positive or negative evaluation of comment
        label_id = torch.argmax(output.logits).item()
        output_one_video.append(label_id)
        # label = model.config.id2label[label_id]
        # print(label_id, label)

      outputs_before.append(output_one_video)

print(len(outputs_before))
outputs_before_df = pd.DataFrame(outputs_before)
outputs_before_df.to_csv('Evaluation_before_finetuning.csv', encoding='utf-8')

###############################################################################################################
#Fine-tune pretrained model
#https://huggingface.co/course/chapter3/1?fw=pt

#Preprocess train and test data
def tokenize(data):
    return tokenizer(data["text"], padding=True, truncation=True, return_tensors='pt')

dataset_tweets_prep = pd.read_csv("Tweets_prep.csv")
dataset_tweets_prep = dataset_tweets_prep.mask(dataset_tweets_prep.eq('None')).dropna() # remove comments where None is stored, otherwise tokenizer throws error

dataset_tweets_prep = datasets.Dataset(pa.Table.from_pandas(dataset_tweets_prep))
training_data2, test_data2 = dataset_tweets_prep.train_test_split(test_size=0.2).values()
dataset_tweets_prep = datasets.DatasetDict({"train":training_data2,"test":test_data2})

tokenized_datasets_tweets_prep = dataset_tweets_prep.map(tokenize, batched=True)
tokenized_datasets_tweets_prep = tokenized_datasets_tweets_prep.remove_columns(['__index_level_0__', 'text', 'Unnamed: 0'])
tokenized_datasets_tweets_prep.set_format("torch")

collator = DataCollatorWithPadding(tokenizer=tokenizer)

#Create Dataloader
batch_size = 8
train_dataloader = torch.utils.data.DataLoader(tokenized_datasets_tweets_prep["train"], batch_size=batch_size, collate_fn=collator)
test_dataloader = torch.utils.data.DataLoader(tokenized_datasets_tweets_prep["test"], batch_size=batch_size, collate_fn=collator)

#Train loop
def train(dataloader, model, optimizer, batch_size, progress_bar):
  total_loss = 0
  true_positive = 0
  true_negative = 0
  false_positive = 0
  false_negative = 0
  model.train()
  for batch in list(dataloader):
    batch = {k: v.to(device) for k, v in batch.items()}
    preds = model(**batch)
    loss = preds.loss

    #Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    total_loss += loss
    progress_bar.update(1)

    predicted = torch.argmax(preds.logits, -1)
    references= batch["labels"]
    for i in range(len(predicted)):
      if(predicted[i] == 1 and references[i] == 1):
        true_positive += 1
      if(predicted[i] == 1 and references[i] == 0):
        false_negative += 1
      if(predicted[i] == 0 and references[i] == 1):
        false_positive += 1
      if(predicted[i] == 0 and references[i] == 0):
        true_negative += 1  
  loss_value = (total_loss/len(dataloader)).item()
  precision = true_positive/(true_positive + false_positive)
  recall = true_positive/(true_positive + false_negative)
  f_score = 2*precision*recall / (precision + recall)
  accuracy = (true_positive + true_negative)/(len(dataloader)*batch_size)
  print("Training individual - Loss value:", loss_value, "Precision:", precision, 
        "Recall:", recall, "F_score:", f_score, "Accuracy:", accuracy)

#Test loop
def test(dataloader, model, batch_size):
  model.eval()
  total_loss = 0 
  true_positive = 0
  true_negative = 0
  false_positive = 0
  false_negative = 0
  for batch in dataloader:
      batch = {k: v.to(device) for k, v in batch.items()}
      with torch.no_grad():
          preds = model(**batch)
      loss = preds.loss
      total_loss += loss
      predicted = torch.argmax(preds.logits, -1)
      references= batch["labels"]
      for i in range(len(predicted)):
        if(predicted[i] == 1 and references[i] == 1):
          true_positive += 1
        if(predicted[i] == 1 and references[i] == 0):
          false_negative += 1
        if(predicted[i] == 0 and references[i] == 1):
          false_positive += 1
        if(predicted[i] == 0 and references[i] == 0):
          true_negative += 1
  loss_value = (total_loss/len(dataloader)).item()
  precision = true_positive/(true_positive + false_positive)
  recall = true_positive/(true_positive + false_negative)
  f_score = 2*precision*recall / (precision + recall)
  accuracy = (true_positive + true_negative)/(len(dataloader)*batch_size)
  print("Testing individual - Loss value:", loss_value, "Precision:", precision, 
        "Recall:", recall, "F_score:", f_score, "Accuracy:", accuracy)

#Run train and test loop
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
num_epochs = 3
progress_bar = tqdm(range(num_epochs * len(train_dataloader)))

for t in range(num_epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, optimizer, batch_size, progress_bar)
    test(test_dataloader, model, batch_size)

###############################################################################################################
#Calculating output after fine-tuning model

outputs_after = []

with torch.no_grad():
    for video in eval_dataset:
      output_one_video = []
      for comment in video:
        input = tokenizer(comment, padding=True, truncation=True, return_tensors='pt')
        input.to(device)
        output = model(**input)
        #Get positive or negative evaluation of comment
        label_id = torch.argmax(output.logits).item()
        output_one_video.append(label_id)
        # label = model.config.id2label[label_id]
        # print(label_id, label)

      outputs_after.append(output_one_video)

print(len(outputs_after))
outputs_after_df = pd.DataFrame(outputs_after)
outputs_after_df.to_csv('Evaluation_after_finetuning.csv', encoding='utf-8')

###############################################################################################################
# Comparing output before finetuning and afterwards
total_same_eval = 0
total_diff_eval = 0
total_pos_before = 0
total_neg_before = 0
total_pos_after = 0
total_neg_after = 0

for idx1 in range(len(outputs_before)):
  for idx2 in range(len(outputs_before[idx1])):
    if outputs_before[idx1][idx2] == outputs_after[idx1][idx2]:
      total_same_eval += 1
    else:
      print("Comment:", eval_dataset[idx1][idx2], "Before:", outputs_before[idx1][idx2], "After:", outputs_after[idx1][idx2])
      total_diff_eval += 1
    if(outputs_after[idx1][idx2] == 0):
      total_neg_after += 1
    if(outputs_after[idx1][idx2] == 1):
      total_pos_after += 1
    if(outputs_before[idx1][idx2] == 0):
      total_neg_before += 1
    if(outputs_before[idx1][idx2] == 0):
      total_pos_before += 1


print("Total same evaluation:", total_same_eval)
print("Total different evaluation:", total_diff_eval)
print("Before - positive:", total_pos_before, "negative:", total_neg_before)
print("After - positive:", total_pos_after, "negative:", total_neg_after)

###############################################################################################################
# Baseline model from sklearn - sklearn.linear_model.SGDClassifier

#<------------------------------------------------------------------------------
#Fit model with train data
temp_X = tokenized_datasets_tweets_prep["train"]["input_ids"]
y = tokenized_datasets_tweets_prep["train"]["labels"].tolist()
max_len_x = 48

#change type of temp_X from tensor to list and make all entries the same length
X = []
for i in range(len(temp_X)):
  temp_list = temp_X[i].tolist()
  if(len(temp_list) != max_len_x):
    for i in range(max_len_x - len(temp_list)):
     temp_list.append(0)
  X.append(temp_list)

#Implementation of model from https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html#sklearn.linear_model.SGDClassifier
baseline_model = make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000, tol=1e-3))
baseline_model.fit(X, y)

#<------------------------------------------------------------------------------
#Test the baseline model with test data
temp_input = tokenized_datasets_tweets_prep["test"]["input_ids"]
input = []
max_len_x = 48

for i in range(len(temp_input)):
  temp_list = temp_input[i].tolist()
  if(len(temp_list) != max_len_x):
    for i in range(max_len_x - len(temp_list)):
     temp_list.append(0)
  input.append(temp_list)

predicted = baseline_model.predict(input)
print(predicted)
references = tokenized_datasets_tweets_prep["test"]["labels"].tolist()

true_positive = 0
true_negative = 0
false_positive = 0
false_negative = 0
for i in range(len(predicted)):
  if(predicted[i] == 1 and references[i] == 1):
    true_positive += 1
  if(predicted[i] == 1 and references[i] == 0):
    false_positive += 1
  if(predicted[i] == 0 and references[i] == 1):
    false_negative += 1
  if(predicted[i] == 0 and references[i] == 0):
    true_negative += 1

precision = true_positive/(true_positive + false_positive)
recall = true_positive/(true_positive + false_negative)
f_score = 2*precision*recall / (precision + recall)
accuracy = (true_positive + true_negative)/(true_positive + true_negative + false_positive + false_negative)
print("Baseline model - Precision:", precision, "Recall:", recall,
      "F_score:", f_score, "Accuracy:", accuracy)