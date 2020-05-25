import nltk
from nltk.stem.lancaster import  LancasterStemmer
stemmer=LancasterStemmer()

import numpy
import random
import tensorflow
import tflearn
import pickle


import json




with open("intents.json") as file:
    data=json.load(file)

try:
    with open("data.pickle","rb") as f:
        words,labels,training,output=pickle.load(f)
except:
    words=[]
    labels=[]
    docs_x=[]
    docs_y=[]
    for intent in data["intents"]:
        for pattern in intent["patterns"]:

             wrd= nltk.word_tokenize(pattern)
             words.extend(wrd)
             docs_x.append(wrd)
             docs_y.append(intent["tag"])

             if intent["tag"] not in labels:
                 labels.append(intent["tag"])
    words =[stemmer.stem(w.lower()) for w in words if w not in "?" ]
    words=sorted(list(set(words)))

    labels=sorted(labels)

    training=[]
    output=[]

    out_empty=[0 for _ in range(len(labels))]

    for x,doc in enumerate(docs_x):
        bag=[]
        wrd=[stemmer.stem(w)for w in doc]

        for w in words:
            if  w in wrd:
                bag.append(1)
            else:
                bag.append(0)
        output_row=out_empty[:]
        output_row[labels.index(docs_y[x])]=1


        training.append(bag)
        output.append((output_row))

    training=numpy.array(training)
    output=numpy.array( output)

    with open("data.pickle","wb") as f:
        pickle.dump((words,labels,training,output),f)

net=tflearn.input_data(shape=[None,len(training[0])])
net=tflearn.fully_connected(net,8)
net=tflearn.fully_connected(net,8)
net=tflearn.fully_connected(net,len(output[0]),activation="softmax")
net=tflearn.regression(net)


model=tflearn.DNN(net)
try:
    model.load("model1.tflearn")
except:
    model.fit(training,output,n_epoch=1000,batch_size=8,show_metric=True)
    model.save("model1.tflearn")

def bag_of_words(s,words):
    bag=[0 for _ in range(len(words))]

    s_word=nltk.word_tokenize(s)
    s_words=[stemmer.stem(word.lower())for word in s_word]

    for se in s_words:
        for i ,w in enumerate(words):
            if w == se:
                bag[i]=(1)

    return  numpy.array(bag)



def chat():
    print("start talking with the bot!(Type quit to stop)")
    while True:
        imp=input("You :")
        if(imp.lower()=="quit"):
            break

        results = model.predict([bag_of_words(imp,words)])
        results_index=numpy.argmax(results)
        tag=labels[results_index]
        for tg in data["intents"]:
            if tg['tag']==tag:
                response=tg['responses']
        print(random.choice(response))


chat()


