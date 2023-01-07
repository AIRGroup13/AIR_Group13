import torch
import numpy as np
import pandas as pd
import pyarrow as pa

from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, get_scheduler
import datasets

from tqdm.auto import tqdm
from matplotlib.pylab import plt

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
def train(dataloader, model, optimizer, batch_size, progress_bar, scheduler):
  total_loss = 0
  true_positive = 0
  true_negative = 0
  false_positive = 0
  false_negative = 0
  model.train()
  for batch in list(dataloader):
    batch = {index: tensor.to(device) for index, tensor in batch.items()}
    preds = model(**batch)
    loss = preds.loss

    #Backpropagation
    loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

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
  accuracy = (true_positive + true_negative)/(true_positive + true_negative + false_positive + false_negative)
  specificity = true_negative / (true_negative + false_positive)
  print("Training - Loss value:", loss_value, "Precision:", precision, 
        "Recall:", recall, "Specificity:", specificity, "F_score:", f_score, "Accuracy:", accuracy)
  return loss_value, accuracy, precision, recall, f_score, specificity

#Test loop
def test(dataloader, model, batch_size):
  model.eval()
  total_loss = 0 
  true_positive = 0
  true_negative = 0
  false_positive = 0
  false_negative = 0
  for batch in dataloader:
      batch = {index: tensor.to(device) for index, tensor in batch.items()}
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
          false_positive += 1
        if(predicted[i] == 0 and references[i] == 1):
          false_negative += 1
        if(predicted[i] == 0 and references[i] == 0):
          true_negative += 1

  print("TP:", true_positive, "TN:", true_negative, "FN:", false_negative, "FP:", false_positive)
  loss_value = (total_loss/len(dataloader)).item()
  precision = true_positive/(true_positive + false_positive)
  recall = true_positive/(true_positive + false_negative)
  f_score = 2*precision*recall / (precision + recall)
  accuracy = (true_positive + true_negative)/(true_positive + true_negative + false_positive + false_negative)
  specificity = true_negative / (true_negative + false_positive)
  print("Testing - Loss value:", loss_value, "Precision:", precision, 
        "Recall:", recall, "Specificity:", specificity, "F_score:", f_score, "Accuracy:", accuracy)
  return loss_value, accuracy, precision, recall, f_score, specificity

#Run train and test loop
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
num_epochs = 15
scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_training_steps=num_epochs*len(train_dataloader),
    num_warmup_steps=0
)
progress_bar = tqdm(range(num_epochs * len(train_dataloader)))
loss_values_train = []
loss_values_test = []
accuracy_train = []
accuracy_test = []
precision_train = []
precision_test = []
recall_train = []
recall_test = []
f_score_train = []
f_score_test = []
specificity_train = []
specificity_test = []

for t in range(num_epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    loss_value, accuracy, precision, recall, f_score, specificity = train(train_dataloader, model, optimizer, batch_size, progress_bar, scheduler)
    loss_values_train.append(loss_value)
    accuracy_train.append(accuracy)
    precision_train.append(precision)
    recall_train.append(recall)
    f_score_train.append(f_score)
    specificity_train.append(specificity)
    loss_value, accuracy, precision, recall, f_score, specificity = test(test_dataloader, model, batch_size)
    loss_values_test.append(loss_value)
    accuracy_test.append(accuracy)
    precision_test.append(precision)
    recall_test.append(recall)
    f_score_test.append(f_score)
    specificity_test.append(specificity)

#Plot results
#Plot training and test loss
plt.plot(range(1, num_epochs+1), loss_values_train, label='Training Loss')
plt.plot(range(1, num_epochs+1), loss_values_test, label='Test Loss')
plt.title('Training and Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='best')
plt.show()

#Plot training and test accuracy
plt.plot(range(1, num_epochs + 1), accuracy_train, label='Training Accuracy')
plt.plot(range(1, num_epochs + 1), accuracy_test, label='Test Accuracy')
plt.title('Training and Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.show()

#Plot training and test precision
plt.plot(range(1, num_epochs + 1), precision_train, label='Training Precision')
plt.plot(range(1, num_epochs + 1), precision_test, label='Test Precision')
plt.title('Training and Test Precision')
plt.xlabel('Epochs')
plt.ylabel('Precision')
plt.legend(loc='best')
plt.show()

#Plot training and test recall
plt.plot(range(1, num_epochs + 1), recall_train, label='Training Recall')
plt.plot(range(1, num_epochs + 1), recall_test, label='Test Recall')
plt.title('Training and Test Recall')
plt.xlabel('Epochs')
plt.ylabel('Recall')
plt.legend(loc='best')
plt.show()

#Plot training and test f1
plt.plot(range(1, num_epochs + 1), f_score_train, label='Training F1')
plt.plot(range(1, num_epochs + 1), f_score_test, label='Test F1')
plt.title('Training and Test F1')
plt.xlabel('Epochs')
plt.ylabel('F1')
plt.legend(loc='best')
plt.show()

#Plot training and test specificity
plt.plot(range(1, num_epochs + 1), specificity_train, label='Training Specificity')
plt.plot(range(1, num_epochs + 1), specificity_test, label='Test Specificity')
plt.title('Training and Test Specificity')
plt.xlabel('Epochs')
plt.ylabel('Specificity')
plt.legend(loc='best')
plt.show()

#Output
# Epoch 1
# -------------------------------
# Training - Loss value: 0.5362582802772522 Precision: 0.8693065506653019 Recall: 0.8345513725971873 Specificity: 0.6376374600922313 F_score: 0.8515744947516841 Accuracy: 0.7839058437115095
# TP: 3551 TN: 900 FN: 443 FP: 587
# Testing - Loss value: 0.42393428087234497 Precision: 0.8581440309328178 Recall: 0.8890836254381572 Specificity: 0.605245460659045 F_score: 0.8733398917855386 Accuracy: 0.8120780879401569
# Epoch 2
# -------------------------------
# Training - Loss value: 0.41354262828826904 Precision: 0.9099923234390993 Recall: 0.8465246369911926 Specificity: 0.7250341997264022 F_score: 0.87711185102972 Accuracy: 0.8181652296884266
# TP: 3560 TN: 977 FN: 434 FP: 510
# Testing - Loss value: 0.3939891457557678 Precision: 0.8746928746928747 Recall: 0.8913370055082624 Specificity: 0.6570275722932079 F_score: 0.8829365079365079 Accuracy: 0.8277686553548622
# Epoch 3
# -------------------------------
# Training - Loss value: 0.3832937479019165 Precision: 0.9165813715455476 Recall: 0.8638610876642951 Specificity: 0.7555763823805061 F_score: 0.8894406853311814 Accuracy: 0.8375074129829844
# TP: 3579 TN: 1023 FN: 415 FP: 464
# Testing - Loss value: 0.3813185691833496 Precision: 0.8852337373237695 Recall: 0.8960941412118177 Specificity: 0.6879623402824478 F_score: 0.8906308324001494 Accuracy: 0.8396278051450465
# Epoch 4
# -------------------------------
# Training - Loss value: 0.3662773370742798 Precision: 0.923810133060389 Recall: 0.8698873561833624 Specificity: 0.7761278195488722 F_score: 0.8960382216982595 Accuracy: 0.8471328862734364
# TP: 3568 TN: 1041 FN: 426 FP: 446
# Testing - Loss value: 0.3756815493106842 Precision: 0.8888888888888888 Recall: 0.8933400100150225 Specificity: 0.7000672494956288 F_score: 0.8911088911088911 Accuracy: 0.8409049443532202
# Epoch 5
# -------------------------------
# Training - Loss value: 0.3510752320289612 Precision: 0.9287998976458547 Recall: 0.8755351866369173 Specificity: 0.7914949419258149 F_score: 0.901381344094366 Accuracy: 0.8550704803612974
# TP: 3591 TN: 1047 FN: 403 FP: 440
# Testing - Loss value: 0.3722190260887146 Precision: 0.8908459439345076 Recall: 0.8990986479719579 Specificity: 0.7041022192333557 F_score: 0.8949532710280375 Accuracy: 0.8461959496442255
# Epoch 6
# -------------------------------
# Training - Loss value: 0.3395426273345947 Precision: 0.9329580348004094 Recall: 0.8801979600458688 Specificity: 0.804185351270553 F_score: 0.9058103785596721 Accuracy: 0.8616395237443547
# TP: 3603 TN: 1045 FN: 391 FP: 442
# Testing - Loss value: 0.37122294306755066 Precision: 0.8907292954264524 Recall: 0.9021031547320981 Specificity: 0.7027572293207801 F_score: 0.8963801467844259 Accuracy: 0.8480204342273308
# Epoch 7
# -------------------------------
# Training - Loss value: 0.32896530628204346 Precision: 0.9340455475946776 Recall: 0.8846410178733717 Specificity: 0.8096381093057607 F_score: 0.9086722469427763 Accuracy: 0.8661101227133798
# TP: 3612 TN: 1043 FN: 382 FP: 444
# Testing - Loss value: 0.37064385414123535 Precision: 0.8905325443786982 Recall: 0.9043565348022033 Specificity: 0.7014122394082044 F_score: 0.8973913043478261 Accuracy: 0.8492975734355045
# Epoch 8
# -------------------------------
# Training - Loss value: 0.32135316729545593 Precision: 0.9376279426816786 Recall: 0.8874962155616106 Specificity: 0.8196448390677026 F_score: 0.9118735807384825 Accuracy: 0.8707631951097121
# TP: 3619 TN: 1038 FN: 375 FP: 449
# Testing - Loss value: 0.3709103763103485 Precision: 0.8896263520157326 Recall: 0.9061091637456185 Specificity: 0.6980497646267653 F_score: 0.8977921111386753 Accuracy: 0.8496624703521255
# Epoch 9
# -------------------------------
# Training - Loss value: 0.3119269013404846 Precision: 0.9385875127942682 Recall: 0.8915896937287312 Specificity: 0.8243366880146387 F_score: 0.9144851657940664 Accuracy: 0.8748232288672962
# TP: 3606 TN: 1049 FN: 388 FP: 438
# Testing - Loss value: 0.37333810329437256 Precision: 0.8916913946587537 Recall: 0.9028542814221332 Specificity: 0.7054472091459314 F_score: 0.8972381189350584 Accuracy: 0.8492975734355045
# Epoch 10
# -------------------------------
# Training - Loss value: 0.3071236312389374 Precision: 0.9410184237461617 Recall: 0.8930305973773677 Specificity: 0.8307946412185722 F_score: 0.9163967106902567 Accuracy: 0.8775603302769034
# TP: 3615 TN: 1041 FN: 379 FP: 446
# Testing - Loss value: 0.37315088510513306 Precision: 0.8901748337847821 Recall: 0.9051076614922383 Specificity: 0.7000672494956288 F_score: 0.8975791433891992 Accuracy: 0.849480021893815
# Epoch 11
# -------------------------------
# Training - Loss value: 0.30141934752464294 Precision: 0.9403147389969294 Recall: 0.8967179111761835 Specificity: 0.8312533912099838 F_score: 0.9179990007494379 Accuracy: 0.8802061949728571
# TP: 3611 TN: 1041 FN: 383 FP: 446
# Testing - Loss value: 0.37494826316833496 Precision: 0.8900665516391423 Recall: 0.9041061592388583 Specificity: 0.7000672494956288 F_score: 0.8970314246677432 Accuracy: 0.8487502280605729
# Epoch 12
# -------------------------------
# Training - Loss value: 0.2964521050453186 Precision: 0.9438331627430911 Recall: 0.8972815179711732 Specificity: 0.8397225264695144 F_score: 0.9199688230709275 Accuracy: 0.8828976780256376
# TP: 3612 TN: 1045 FN: 382 FP: 442
# Testing - Loss value: 0.3764781653881073 Precision: 0.8909718796250616 Recall: 0.9043565348022033 Specificity: 0.7027572293207801 F_score: 0.8976143141153081 Accuracy: 0.8496624703521255
# Epoch 13
# -------------------------------
# Training - Loss value: 0.29120156168937683 Precision: 0.9442809621289662 Recall: 0.9011049386484341 Specificity: 0.8427797833935018 F_score: 0.9221878611813951 Accuracy: 0.8863646731444733
# TP: 3615 TN: 1040 FN: 379 FP: 447
# Testing - Loss value: 0.3780309557914734 Precision: 0.8899556868537666 Recall: 0.9051076614922383 Specificity: 0.699394754539341 F_score: 0.8974677259185699 Accuracy: 0.8492975734355045
# Epoch 14
# -------------------------------
# Training - Loss value: 0.2908450663089752 Precision: 0.9442169907881269 Recall: 0.9006590187942397 Specificity: 0.8424001445870233 F_score: 0.9219237976264835 Accuracy: 0.8859541079330322
# TP: 3612 TN: 1039 FN: 382 FP: 448
# Testing - Loss value: 0.3788953125476837 Precision: 0.8896551724137931 Recall: 0.9043565348022033 Specificity: 0.6987222595830531 F_score: 0.8969456170846785 Accuracy: 0.8485677796022624
# Epoch 15
# -------------------------------
# Training - Loss value: 0.2888408303260803 Precision: 0.9433213920163767 Recall: 0.9011794903135122 Specificity: 0.8405901403382512 F_score: 0.9217690264103767 Accuracy: 0.8858172528625519
# TP: 3621 TN: 1037 FN: 373 FP: 450
# Testing - Loss value: 0.3788900375366211 Precision: 0.8894620486366986 Recall: 0.9066099148723085 Specificity: 0.6973772696704774 F_score: 0.8979541227526349 Accuracy: 0.849844918810436


###############################################################################################################
#Calculating labels for youtube comments after fine-tuning model

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

#Output:
# Total same evaluation: 11397
# Total different evaluation: 4153
# Before - positive: 5306 negative: 5306
# After - positive: 14333 negative: 1217

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

print("TP:", true_positive, "TN:", true_negative, "FN:", false_negative, "FP:", false_positive)
precision = true_positive/(true_positive + false_positive)
recall = true_positive/(true_positive + false_negative)
f_score = 2*precision*recall / (precision + recall)
accuracy = (true_positive + true_negative)/(true_positive + true_negative + false_positive + false_negative)
specificity = true_negative / (true_negative + false_positive)
print("Baseline model - Precision:", precision, "Recall:", recall,
      "F_score:", f_score, "Specificity:", specificity, "Accuracy:", accuracy)

#Output:
# TP: 3992 TN: 3 FN: 2 FP: 1484
# Baseline model - Precision: 0.7289992695398101 Recall: 0.99949924887331 F_score: 0.8430834213305174 Specificity: 0.0020174848688634837 Accuracy: 0.7288815909505565

############################################################################
#Loop used for checking if values from basemodel are always kinda the same - and they are
# for t in range(num_epochs):
#     print(f"Epoch {t+1}\n-------------------------------")
#     #Training
#     baseline_model.fit(X, y)
#     #Testing
#     predicted = baseline_model.predict(input)
#     true_positive = 0
#     true_negative = 0
#     false_positive = 0
#     false_negative = 0
#     for i in range(len(predicted)):
#       if(predicted[i] == 1 and references[i] == 1):
#         true_positive += 1
#       if(predicted[i] == 1 and references[i] == 0):
#         false_positive += 1
#       if(predicted[i] == 0 and references[i] == 1):
#         false_negative += 1
#       if(predicted[i] == 0 and references[i] == 0):
#         true_negative += 1

#     print("TP:", true_positive, "TN:", true_negative, "FN:", false_negative, "FP:", false_positive)
#     precision = true_positive/(true_positive + false_positive)
#     recall = true_positive/(true_positive + false_negative)
#     f_score = 2*precision*recall / (precision + recall)
#     accuracy = (true_positive + true_negative)/(true_positive + true_negative + false_positive + false_negative)
#     print("Baseline model - Precision:", precision, "Recall:", recall,
#           "F_score:", f_score, "Accuracy:", accuracy)