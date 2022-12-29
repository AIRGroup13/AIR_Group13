import torch
import evaluate
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds


from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, DataCollatorWithPadding, TrainingArguments, AdamW, get_scheduler
from datasets import load_dataset, Dataset
from enum import Enum
import datasets

texts = ["Really good video!", "Wasn't that bad", "Thank you", "She drives a green car", "TOP",
        "It was the worst", "A waste of time!", "Informative video and well explained."]

#TODO
# - evaluation part of Trainer I think is wrong, should not be evaluate.load("glue", "mrpc") because glue/mrpc is the dataset used in the tutorial from huggingface.co
#   I changed it now to "bert_score" - hope it works that way and with bertscore also precision and recall should be able to be calculated
# - calculate precision and recall

####################################################################################################################
#BERT - has 5 classes -  5 is very positive - 1 is very negative
# class Label(Enum):
#     POSITIVE = 1
#     NEGATIVE = 2
#     NEUTRAL = 3

# tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
# model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

# with torch.no_grad():
#     input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
#     output = model(**input)
#     for sequence in output.logits:
#         label_id = torch.argmax(sequence).item()
#         label = model.config.id2label[label_id]
#         print(label_id, label)
#         if(label_id == 0 or label_id == 1):
#             print(Label.NEGATIVE.name)
#         elif(label_id == 2):
#             print(Label.NEUTRAL.name)
#         if(label_id == 3 or label_id == 4):
            # print(Label.POSITIVE.name)
####################################################################################################################
#Distilbert
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

###############################################################################################################
#Fine-tune pretrained model
#https://huggingface.co/course/chapter3/1?fw=pt

def tokenize(data):
    return tokenizer(data["text"], padding=True, truncation=True, return_tensors='pt')

dataset = pd.read_csv("Tweets.csv")

dataset = dataset.drop(columns=['textID', 'selected_text'])
dataset = dataset.rename(columns={"sentiment": "labels"})
dataset = dataset.mask(dataset.eq('None')).dropna() # remove comments where None is stored, otherwise tokenizer throws error

dataset = Dataset(pa.Table.from_pandas(dataset))
training_data, test_data = dataset.train_test_split(test_size=0.2).values()
dataset = datasets.DatasetDict({"train":training_data,"test":test_data})

tokenized_datasets = dataset.map(tokenize, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(['__index_level_0__', 'text'])
tokenized_datasets.set_format("torch")

collator = DataCollatorWithPadding(tokenizer=tokenizer)

def compute_metrics(eval_preds):
    #metric = evaluate.load("glue", "mrpc")
    metric = evaluate.load("bertscore")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
print(device)

trainer = Trainer(
    model,
    TrainingArguments("trainer_without_h", evaluation_strategy="epoch"),
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# trainer.train()

###############################################################################################################
#Calculating output with comments from array text
#will be changed to comments from youtube channel

with torch.no_grad():
    input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    output = model(**input)
    for sequence in output.logits:
        label_id = torch.argmax(sequence).item()
        label = model.config.id2label[label_id]
        print(label_id, label)

##################################################################################################################
#DATASET yelp_reviews
# dataset = load_dataset("yelp_review_full")
# tokenized_datasets = dataset.map(tokenize, batched=True)
# tokenized_datasets = tokenized_datasets.remove_columns(["text"]).rename_column("label", "labels")#
# tokenized_datasets.set_format("torch")
# collator = DataCollatorWithPadding(tokenizer=tokenizer)

##################################################################################################################
#Tried to implement the Trainer by hand - does not really work now, but it would look better

# #Dataloader
# batch_size = 25
# train_dataloader = torch.utils.data.DataLoader(tokenized_datasets["train"], batch_size=batch_size, collate_fn=collator)
# test_dataloader = torch.utils.data.DataLoader(tokenized_datasets["test"], batch_size=batch_size, collate_fn=collator)
# optimizer = AdamW(model.parameters(), lr=5e-5)

# num_epochs = 3
# num_training_steps = num_epochs * len(train_dataloader)
# lr_scheduler = get_scheduler(
#     "linear",
#     optimizer=optimizer,
#     num_training_steps=num_training_steps,
# )

# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# model.to(device)

# # metric = evaluate.load("glue")
# for epoch in range(num_epochs):
#     model.train()
#     for batch in train_dataloader:
#         outputs = model(**batch)
#         loss = outputs.loss
#         loss.backward()

#         optimizer.step()
#         lr_scheduler.step()
#         optimizer.zero_grad()

    # model.eval()
    # for batch in test_dataloader:
    #     with torch.no_grad():
    #         outputs = model(**batch)

    #     logits = outputs.logits
    #     predictions = torch.argmax(logits, dim=-1)
    #     metric.add_batch(predictions=predictions, references=batch["labels"])

    # metric.compute()

