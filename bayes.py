import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_regression, make_classification, make_blobs
from sklearn.linear_model import LinearRegression


def bayesClassfr(dataset):
    X = np.zeros((len(dataset), 3))
    y = np.zeros(len(dataset))

    for i in range(len(dataset)):
        X[i] = np.array([dataset[i][1], dataset[i][2], dataset[i][3]])
        y[i] = 1 if dataset[i][4] > 0.75 else 0
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    from sklearn.naive_bayes import GaussianNB
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    plt.plot(X_test, y_pred, 'ro')
    
    plt.savefig('bayespred.png')
    plt.plot(X_test, y_test, 'ro')
    plt.savefig('bayesacc.png')
    plt.show()
    from sklearn.metrics import confusion_matrix, accuracy_score
    ac = accuracy_score(y_test,y_pred)
    cm = confusion_matrix(y_test, y_pred)
    print(ac)
    print(cm)

def calcNegPercent(comments):
    sum_neg = 0
    for comment in comments:
        if comment == 0:
            sum_neg = sum_neg+1
    return sum_neg/len(comments)

def linReg(dataset):
    X = np.zeros((len(dataset), 3))
    y = np.zeros(len(dataset))

    for i in range(len(dataset)):
        X[i] = np.array([dataset[i][1], dataset[i][2], dataset[i][3]])
        y[i] = dataset[i][4]
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
    # y = 1 * x_0 + 2 * x_1 + 3

    reg = LinearRegression().fit(X_train, y_train)
    print(reg.score(X_test, y_test))
    y_pred = reg.predict(X_test)
    plt.plot(X_test, y_pred,'ro')
    plt.savefig('linreg.png')
    plt.plot(X_test, y_test,'ro')
    plt.savefig('linregacc.png')
    plt.show()

def main():
    df = pd.read_csv("Evaluation_after_finetuning.csv")
    dataset = []
    for row in df.iterrows():
        for comment in row:
            if(type(comment) == int):
                continue
            video = [int(item) for item in comment if not(pd.isnull(item)) == True] #remove NaN comments
            for entry in video:
                if type(entry) is not str: 
                    video.remove(entry) #remove column numbers from dataFrame
            dataset.append(video)
    percentList = []
    sum = 0
    for video in dataset:
        for i in range(len(video)):
            sum = sum + video[i]
        percentList.append(sum/len(video))
        sum = 0
    
    for video in range(len(dataset)):
        if(video == 0):
            dataset[video] = [video, -1, -1, -1, percentList[video]]
        elif(video == 1):
            dataset[video] = [video, percentList[video-1], -1, -1, percentList[video]]
        elif(video == 2):
            dataset[video] = [video, percentList[video-1], percentList[video-2], -1, percentList[video]]
        else:
            dataset[video] = [video, percentList[video-1], percentList[video-2], percentList[video-3], percentList[video]]
        
    print(dataset)

    bayesClassfr(dataset)
    linReg(dataset)


    #create dataset with view
    # Video index | Percent of Neg Comm Video Before | Percent of Neg Comm 2Videos Before | Percent of Neg Comm 3Videos Before | Percent of Neg Comm Video actual
    #              ------------------------------------------------- X values -------------------------------------------------  ----------- y value ------------


#<-------------------------------------------

if __name__ == "__main__":
    main()