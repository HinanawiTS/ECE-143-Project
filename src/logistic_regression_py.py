import numpy as np 
import pandas as pd
import json
from mplsoccer.pitch import Pitch, VerticalPitch


path = "C:/Users/brand/desktop/events/events_England.json" 

with open(path) as f:
    data = json.load(f)

train = pd.DataFrame(data)


path2 = "C:/Users/brand/desktop/players.json" 


with open(path2) as f:
    play = json.load(f)

players = pd.DataFrame(play)

lst = ['events_France.json','events_Germany.json','events_Italy.json','events_Spain.json']
pathway = "C:/Users/brand/desktop/events/"

for country in lst:
    with open(pathway + country) as f:
        datal = json.load(f)
        tl = pd.DataFrame(datal)
        train = pd.concat([train,tl],ignore_index=True)




#pd.unique(train['subEventName'])
shots = train[train['subEventName'] == 'Shot']

print(len(shots))

shots_model = pd.DataFrame(columns=["Goal","X","Y"], dtype=object)

for i,shot in shots.iterrows():
    
    
    shots_model.at[i,'Header'] = 0
    for tag in shot['tags']:
        if tag['id'] == 403:
            shots_model.at[i,'Header'] = 1
    
    
    #take distance from center of goal at y = 50, x position of goal is always 100
    shots_model.at[i,'X'] = 100-shot['positions'][0]['x']
    shots_model.at[i,'Y'] = shot['positions'][0]['y']
    shots_model.at[i,'C'] = abs(shot['positions'][0]['y'] - 50)
        
    #distance in meters
        
    x = shots_model.at[i,'X']* 105/100
    y = shots_model.at[i,'C']* 65/100
    shots_model.at[i,'Distance'] = np.sqrt(x**2 + y**2)
        
    angle = np.arctan(7.32 * x / (x**2 + y**2 - (7.32/2)**2))
        
    if angle < 0:
        angle = np.pi + angle
        
    shots_model.at[i,'Angle'] = angle
        
    #goal check
    shots_model.at[i,'Goal'] = 0
    shots_model.at[i,'Counter Attack'] = 0
    shots_model.at[i, 'Blocked'] = 0
    shots_model.at[i, 'Right Foot'] = 0
    shots_model.at[i,'wyId'] = shot['playerId']
    
    if shot['matchPeriod'] == '1H':
        shots_model.at[i, 'First Half'] = 1
    
    else:
        shots_model.at[i,'First Half'] = 0
        
    for tags in shot['tags']:
        if tags['id'] == 101:
            shots_model.at[i,'Goal'] = 1
            
        if tags['id'] == 1901:
            shots_model.at[i, 'Counter Attack'] = 1
        
        if tags['id'] == 2101:
            shots_model.at[i, 'Blocked'] = 1
        
        if tags['id'] == 402:
            shots_model.at[i, 'Right Foot'] = 1
            
            
        
shots_model['angle_degrees'] = shots_model['Angle'] * 180 / np.pi

shots_model = shots_model.merge(players, left_on = 'wyId' , right_on = 'wyId')

for i,shot in shots_model.iterrows():
    shots_model.at[i, 'strong foot'] = 0
    
    if shot['Right Foot'] == 1:
        if shot['foot'] == 'right':
            shots_model.at[i, 'strong foot'] = 1
    
    elif shot['Right Foot'] == 0:
        if shot['foot'] == 'left':
            shots_model.at[i, 'strong foot'] = 1


pitch = Pitch(pitch_color ='black', line_color = 'white', stripe=False,pitch_type='wyscout')

fig,ax = pitch.draw(figsize=(10,8))

df = shots_model.loc[shots_model['Goal'] == 1]
xpos = df["X"]
ypos = df["Y"]

df_nongoals = shots_model.loc[shots_model['Goal'] == 0]
xpos2 = df_nongoals["X"]
ypos2 = df_nongoals["Y"]



head = shots_model[shots_model['Header'] == 1]
counter = shots_model[shots_model['Counter Attack'] == 1]
strong = shots_model[shots_model['strong foot'] == 1]
first = shots_model[shots_model['First Half'] == 1]
head_df = head.loc[head['Goal'] == 1]
strong_goal = strong.loc[strong['Goal'] == 1]

headed_goals = len(head_df)


from sklearn.model_selection import train_test_split

X_full = shots_model[["Header","Distance", "Angle","Counter Attack","strong foot", "First Half"]]
y_full = shots_model[["Goal"]]
y_full['Goal'] = y_full['Goal'].astype(int)
print(X_full.head())

X_train,X_test,y_train,y_test = train_test_split(X_full,y_full,test_size = 0.15,random_state=2)

X_train,X_val,y_train,y_val = train_test_split(X_train,y_train,test_size = 0.176, random_state=2)


y_val = np.array(y_val)
yval = [i[0] for i in y_val]
y_test = np.array(y_test)
ytest = [i[0] for i in y_test]
y_train = np.array(y_train)
ytrain = [i[0] for i in y_train]
print(len(y_train))


from sklearn.linear_model import LogisticRegression


lambdas = [0.1**i for i in range(-1,5)]
accs = []

for i in lambdas:
    model = LogisticRegression(C=i).fit(X_train,y_train)
    predictions = model.predict(X_val)
    
    correct = [(p==l) for (p,l) in zip(predictions,yval)]
    
    accuracy = sum(correct)/len(correct)
    accs.append(accuracy)
    
    
    
    
    
print(accs)   


import matplotlib.pyplot as plt


lambs = [1,.01,.001,.0001,.00001]
f1 = []
prec = []
rec = []
ber = []
acc = []
for i in lambs:
    log = LogisticRegression(C=i).fit(X_train,ytrain)
    predictions = log.predict(X_val)
    
    TP = sum([(p and l) for (p,l) in zip(predictions, yval)])
    FP = sum([(p and not l) for (p,l) in zip(predictions, yval)])
    TN = sum([(not p and not l) for (p,l) in zip(predictions, yval)])
    FN = sum([(not p and l) for (p,l) in zip(predictions, yval)])
    
    accuracy = (TP + TN)/(TP + FP + TN + FN)
    
    TPR = TP/(TP+FN)
    TNR = TN/(TN+FP)
    
    BER = 1 - .5 * (TPR + TNR)
    
    precision = TP/(TP+FP)
    recall = TP/(TP + FN)
    
    F1 = 2 * (precision*recall)/(precision + recall)
    
    f1.append(F1)
    prec.append(precision)
    rec.append(recall)
    ber.append(BER)
    acc.append(accuracy)


plt.xlabel('Regularization Parameter')
plt.xscale('log')
plt.ylabel('Score')
plt.title('Classification Metrics vs Regularization Parameters: Logistic Regression')
plt.plot(lambs,f1,label='F1')
plt.plot(lambs,prec,label='Precision')
plt.plot(lambs,rec,label='Recall')
plt.plot(lambs,ber,label='Balanced Error Rate')
plt.plot(lambs,acc,label = 'Total Accuracy')
plt.legend(loc=2)
plt.show()



X_val = np.array(X_val)

bestlog = LogisticRegression(C=1,class_weight='balanced').fit(X_train,ytrain)
probs = bestlog.predict_proba(X_val)
prediction = bestlog.predict(X_val)

probY = list(zip([p[1] for p in probs], [p[1] > 0.5 for p in probs], yval))

probY.sort(reverse=True)
print(probY[0:30])

test_predictions = bestlog.predict(X_test)

TP = sum([(p and l) for (p,l) in zip(test_predictions, ytest)])
FP = sum([(p and not l) for (p,l) in zip(test_predictions, ytest)])
TN = sum([(not p and not l) for (p,l) in zip(test_predictions, ytest)])
FN = sum([(not p and l) for (p,l) in zip(test_predictions, ytest)])
    
TPR = TP/(TP+FN)
TNR = TN/(TN+FP)
    
BER = 1 - .5 * (TPR + TNR)
    
precision = TP/(TP+FP)
recall = TP/(TP + FN)
    
F1 = 2 * (precision*recall)/(precision + recall)

accuracy = (TP + TN)/(TP + FP + TN + FN)
    
print("Balanced Error rate: " + str(BER))
print("Precision: " + str(precision))
print('Recall: ' + str(recall))
print('F1 score: ' + str(F1))
print('Total Classification Accuracy: ' + str(accuracy))
print()

print(bestlog.coef_)


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from yellowbrick.regressor import ResidualsPlot


xtestprobs = bestlog.predict_proba(X_test)[:,1]

Y = []
X = []
count1 = 0
count2 = 0
for i in range(len(xtestprobs)):
    count1 += xtestprobs[i]
    count2 += ytest[i]
    
    
    X.append(count1)
    Y.append(count2)
    count1 = 0
    count2 = 0

print(Y[0:10])
X = np.array(X)
X = X.reshape(-1,1)



lin = LinearRegression().fit(X,Y)

print(lin.score(X,Y))

lpreds = lin.predict(X)
print('Coeffs')
print(lin.coef_)

print(mean_squared_error(Y,lpreds))
print(r2_score(Y,lpreds))


visualizer = ResidualsPlot(lin)
visualizer.fit(X,Y)
visualizer.score(X,lpreds)
visualizer.show()
