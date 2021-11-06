from matplotlib import pyplot as plt

ranges = np.arange(1,40)
d = {n:[0,0] for n in ranges}

for index,row in shots_model.iterrows():
    
    dist = round(row['Distance'])
    if dist < 39:
        d[dist][1] += 1
        
        if row['Goal'] == 1:
            d[dist][0] +=1
    
    else:
        d[39][1] +=1
            
        if row['Goal'] == 1:
            d[39][0] += 1

#print(d)
yaxis = []
for i in range(1,40):
    d[i] = d[i][0]/d[i][1]
    yaxis.append(d[i])

#print(yaxis)


plt.scatter(ranges,yaxis,cmap = 'spectral')
plt.grid()
plt.xlabel('Distance')
plt.ylabel('Probability')
plt.title('Distance vs Goal Probability')
plt.show()

angles = np.arange(0,150,5)
angles_dict = {n:[0,0] for n in range(30)}



for index,row in shots_model.iterrows():
    
    ang = round(row['angle_degrees'] / 5)
    
    angles_dict[ang][1] += 1
    
    if row['Goal'] == 1:
        angles_dict[ang][0] += 1

print(angles_dict)
yaxis2 = []
xaxis2 = []
count = 0
for i in range(30):
    if angles_dict[i][1] > 0:
        yaxis2.append(angles_dict[i][0]/angles_dict[i][1])
        xaxis2.append(count)
        count += 5
    
    else:
        count += 5

plt.scatter(xaxis2,yaxis2,c = 'r')
plt.grid()
plt.xlabel('Angle(degrees)')
plt.ylabel('Probability')
plt.title('Angle vs Goal Probability')
plt.show()