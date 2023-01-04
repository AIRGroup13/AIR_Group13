import torch
import pandas as pd
import pyarrow as pa
import datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, AdamW
from tqdm.auto import tqdm

texts = ["Really good video!", "Wasn't that bad", "Thank you", "She drives a green car", "TOP",
        "It was the worst", "A waste of time!", "Informative video and well explained."]

#Get preprocessed comments
df = pd.read_csv("Comments_prep.csv")
dataset = []
for row in df.iterrows():
    one_video = []
    for comment in row:
      if(type(comment) == int):
        continue
      temp_list = [item for item in comment if not(pd.isnull(item)) == True] #remove NaN comments
      for entry in temp_list:
        if type(entry) is not str: 
          temp_list.remove(entry) #remove column numbers from dataFrame
    dataset.append(temp_list)
print(type(dataset), len(dataset))    

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
#Calculating output with comments from array text before fine-tuning model
#will be changed to comments from youtube channel

with torch.no_grad():
    input = tokenizer(dataset, padding=True, truncation=True, is_split_into_words=True, return_tensors='pt')
    input.to(device)
    output = model(**input)
    for sequence in output.logits:
        label_id = torch.argmax(sequence).item()
        label = model.config.id2label[label_id]
        print(label_id, label)

###############################################################################################################
#Fine-tune pretrained model
#https://huggingface.co/course/chapter3/1?fw=pt

#Preprocess train and test data
def tokenize(data):
    return tokenizer(data["text"], padding=True, truncation=True, return_tensors='pt')

dataset = pd.read_csv("Tweets.csv")

dataset = dataset.drop(columns=['textID', 'selected_text'])
dataset = dataset.rename(columns={"sentiment": "labels"})
dataset = dataset.mask(dataset.eq('None')).dropna() # remove comments where None is stored, otherwise tokenizer throws error

dataset = datasets.Dataset(pa.Table.from_pandas(dataset))
training_data, test_data = dataset.train_test_split(test_size=0.2).values()
dataset = datasets.DatasetDict({"train":training_data,"test":test_data})

tokenized_datasets = dataset.map(tokenize, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(['__index_level_0__', 'text'])
tokenized_datasets.set_format("torch")
print(tokenized_datasets)

collator = DataCollatorWithPadding(tokenizer=tokenizer)

#Create Dataloader
batch_size = 8
train_dataloader = torch.utils.data.DataLoader(tokenized_datasets["train"], batch_size=batch_size, collate_fn=collator)
test_dataloader = torch.utils.data.DataLoader(tokenized_datasets["test"], batch_size=batch_size, collate_fn=collator)

#Run train and test loop
optimizer = AdamW(model.parameters(), lr=5e-5)
num_epochs = 3
progress_bar = tqdm(range(num_epochs * len(train_dataloader)))

for t in range(num_epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    #Train loop
    total_loss = 0
    model.train()
    for batch in list(train_dataloader):
      batch = {k: v.to(device) for k, v in batch.items()}
      preds = model(**batch)
      loss = preds.loss

      #Backpropagation
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      total_loss += loss
      progress_bar.update(1)
    print("Training - Loss value:", (total_loss/len(train_dataloader)).item())

    #Test loop
    model.eval()
    total_loss = 0 
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    for batch in test_dataloader:
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

    precision = true_positive/(true_positive + false_positive)
    recall = true_positive/(true_positive + false_negative)
    f_score = 2*precision*recall / (precision + recall)
    accuracy = (true_positive + true_negative)/len(test_data)

    print("Testing - Loss value:", (total_loss/len(test_dataloader)).item(), 
          "Precision:", precision, "Recall:", recall, "F_score:", f_score, "Accuracy:", accuracy)

###############################################################################################################
#Calculating output with comments from array text after fine-tuning model
#will be changed to comments from youtube channel

with torch.no_grad():
    input = tokenizer(dataset, padding=True, truncation=True, is_split_into_words=True, return_tensors='pt')
    input.to(device)
    output = model(**input)
    for sequence in output.logits:
        label_id = torch.argmax(sequence).item()
        label = model.config.id2label[label_id]
        print(label_id, label)

