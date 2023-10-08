import pandas as pd
import tensorflow as tf
import os

df = pd.read_csv('ml-100k/u1.base')
# print(df)
arr=df.values
# print(arr)
rating=[[-1 for _ in range(1682)] for _ in range(943)]
for i in range(len(arr)):
    temp=arr[i][0].split('\t')
    rating[int(temp[0])-1][int(temp[1])-1]=int(temp[2])

def recommed(user,k):
    current_directory = os.path.abspath(os.path.dirname(__file__))
    model_save_path = os.path.join(current_directory, "test_model")
    loaded = tf.saved_model.load(model_save_path)
    print(loaded)
    scores, titles = loaded([str(user)])
    # print(f"Recommendations: {titles[0][:k]}")
    return {'Recommendations':titles[0][:k]}
        

def support(input,k):
    # input={4:5,100:2,50:4,800:3,700:4,300:1,200:4,1:4,2:5,3:5,5:4}
    closest=0
    curr=float('inf')
    for i in range(len(rating)):
        temp=10**8
        for j in range(len(rating[i])):
            if rating[i][j]!=-1 and j+1 in input:
                temp-=abs(input[j+1]-rating[i][j])
        if curr>temp:
            curr=temp
            closest=i
    # print(closest)
    return recommed(closest+1,k)