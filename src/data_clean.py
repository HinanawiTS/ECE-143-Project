import numpy as np 
import pandas as pd
import json
from mplsoccer.pitch import Pitch, VerticalPitch


path = "C:/Users/brand/desktop/events/events_England.json" 

with open(path) as f:
    data = json.load(f)

train = pd.DataFrame(data)
taineng = pd.DataFrame(data)

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


pd.unique(train['subEventName'])
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
    shots_model.at[i,'matchId'] = shot['matchId']
    
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


sc1 = pitch.scatter(xpos, ypos,
                    # size varies between 100 and 1900 (points squared)
                    s=7,
                      # give the markers a charcoal border
                    c='cyan',  # no facecolor for the markers
                    
                    # for other markers types see: https://matplotlib.org/api/markers_api.html
                    marker='o',ax=ax )




pitch2 = VerticalPitch(pitch_color ='white', line_color = 'grey', stripe=False,pitch_type='wyscout')

fig,ax = pitch.draw(figsize=(10,8))

sc2 = pitch.scatter(xpos2, ypos2,
                    # size varies between 100 and 1900 (points squared)
                    s=1,
                      # give the markers a charcoal border
                    c='red',  # no facecolor for the markers
                    
                    # for other markers types see: https://matplotlib.org/api/markers_api.html
                    marker='o',ax=ax )


head = shots_model[shots_model['Header'] == 1]
counter = shots_model[shots_model['Counter Attack'] == 1]
strong = shots_model[shots_model['strong foot'] == 1]
first = shots_model[shots_model['First Half'] == 1]
head_df = head.loc[head['Goal'] == 1]
strong_goal = strong.loc[strong['Goal'] == 1]

headed_goals = len(head_df)


mo_salah = shots_model[shots_model['shortName'] == 'Mohamed Salah']


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


X_full = shots_model[["Header","Distance", "angle_degrees","Counter Attack","strong foot", "First Half"]]
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

bestlog = LogisticRegression(C=.01).fit(X_train,ytrain)

test_preds = bestlog.predict_proba(X_test)[:,1]
print(sum(test_preds))
print(sum(ytest))

mo_shots = mo_salah[["Header","Distance", "angle_degrees","Counter Attack","strong foot", "First Half","Goal","X","Y"]]
mo_goals = mo_shots.loc[mo_shots['Goal'] == 1]
mo_misses = mo_shots.loc[mo_shots['Goal'] == 0]


mo_y_goal = mo_goals[["Goal"]]
mo_y_misses = mo_misses[["Goal"]]

mo_goal_x = mo_goals[["X"]]
mo_goal_y = mo_goals[["Y"]]
mo_miss_x = mo_misses[["X"]]
mo_miss_y = mo_misses[["Y"]]


mo_goals = mo_goals.iloc[:,0:6]
mo_misses = mo_misses.iloc[:,0:6]

mo_goals = np.array(mo_goals)
mo_misses = np.array(mo_misses)


mo_probs_goals = bestlog.predict_proba(mo_goals)[:,1]
mo_probs_misses = bestlog.predict_proba(mo_misses)[:,1]




pitch_mo = Pitch(pitch_color ='green', line_color = 'white', stripe=False,pitch_type='wyscout')

fig,ax = pitch_mo.draw(figsize=(20,16))



sc1 = pitch_mo.scatter(mo_goal_x, mo_goal_y,
                    # size varies between 100 and 1900 (points squared)
                    s=(mo_probs_goals * 1900 + 50 ) ,
                     # give the markers a charcoal border
                    c='red',  # no facecolor for the marker
                    # for other markers types see: https://matplotlib.org/api/markers_api.html
                    marker='football',
                    ax=ax)


sc2 = pitch_mo.scatter(mo_miss_x, mo_miss_y, 
                    # size varies between 100 and 1900 (points squared)
                    s=(mo_probs_misses * 1900 + 50 ) ,
                     # give the markers a charcoal border
                    c='yellow',  # no facecolor for the marker
                    # for other markers types see: https://matplotlib.org/api/markers_api.html
                    marker='football',
                    ax=ax)




richarlison = shots_model[shots_model['shortName'] == 'Richarlison']


rich_shots = richarlison[["Header","Distance", "angle_degrees","Counter Attack","strong foot", "First Half","Goal","X","Y"]]
rich_goals = rich_shots.loc[rich_shots['Goal'] == 1]
rich_misses = rich_shots.loc[rich_shots['Goal'] == 0]


rich_y_goal = rich_goals[["Goal"]]
rich_y_misses = rich_misses[["Goal"]]

rich_goal_x = rich_goals[["X"]]
rich_goal_y = rich_goals[["Y"]]
rich_miss_x = rich_misses[["X"]]
rich_miss_y = rich_misses[["Y"]]


rich_goals = rich_goals.iloc[:,0:6]
rich_misses = rich_misses.iloc[:,0:6]

rich_goals = np.array(rich_goals)
rich_misses = np.array(rich_misses)


rich_probs_goals = bestlog.predict_proba(rich_goals)[:,1]
rich_probs_misses = bestlog.predict_proba(rich_misses)[:,1]


pitch_rich = Pitch(pitch_color ='green', line_color = 'white',pitch_type='wyscout')

fig,ax = pitch_rich.draw(figsize=(20,16))



sc1 = pitch_rich.scatter(rich_goal_x, rich_goal_y,
                    # size varies between 100 and 1900 (points squared)
                    s=(rich_probs_goals * 1900 + 50 ) ,
                     # give the markers a charcoal border
                    c='red',  # no facecolor for the marker
                    # for other markers types see: https://matplotlib.org/api/markers_api.html
                    marker='football',
                    ax=ax)


sc2 = pitch_rich.scatter(rich_miss_x, rich_miss_y, 
                    # size varies between 100 and 1900 (points squared)
                    s=(rich_probs_misses * 1900 + 50 ) ,
                     # give the markers a charcoal border
                    c='yellow',  # no facecolor for the marker
                    # for other markers types see: https://matplotlib.org/api/markers_api.html
                    marker='football',
                    ax=ax)

print((sum(rich_probs_goals) + sum(rich_probs_misses))/90)
print((sum(mo_probs_goals) + sum(mo_probs_misses))/136)



