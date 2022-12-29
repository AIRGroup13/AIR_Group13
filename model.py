import torch

#BERT - has 5 classes -  5 is very positive - 1 is very negative
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

texts = ["Really good video!", "Wasn't that bad", "Thank you", "She drives a green car", "TOP",
        "It was the worst", "A waste of time!", "Informative video and well explained."]

for text in texts:  
    with torch.no_grad():
        input = tokenizer(text, return_tensors='pt')
        output = model(**input)
        label_id = torch.argmax(output.logits).item()
        # print(label_id)
        print(model.config.id2label[label_id])

#DISTILBERT - without neutral   
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

for text in texts:  
    with torch.no_grad():
        input = tokenizer(text, return_tensors='pt')
        output = model(**input)
        label_id = torch.argmax(output.logits).item()
        # print(label_id)
        print(model.config.id2label[label_id])


#BERTWEET - positive, negative and neutral - sometimes different output labels than BERT and DISTILBERT
tokenizer = AutoTokenizer.from_pretrained("finiteautomata/bertweet-base-sentiment-analysis")
model = AutoModelForSequenceClassification.from_pretrained("finiteautomata/bertweet-base-sentiment-analysis")

for text in texts:  
    with torch.no_grad():
        input = tokenizer(text, return_tensors='pt')
        output = model(**input)
        label_id = torch.argmax(output.logits).item()
        # print(label_id)
        print(model.config.id2label[label_id])
