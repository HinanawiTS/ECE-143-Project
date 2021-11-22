import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

import pandas as pd 

import numpy as np 
from mplsoccer.pitch import Pitch, VerticalPitch

def plt_distance_goals(dataset): 
    """ 
    Plot distance vs. Goal Probability. 
    
    :param dataset: input shots dataset to graph 
    :type dataset: Pandas DataFrame
    
    :returns: None 
    :type returns: None 
    
    """ 
    
    assert isinstance(dataset, pd.DataFrame)
    assert len(dataset) > 0
    
    plt.figure(figsize = (7, 5))
    dataset["rounded_distance"] = dataset.apply(lambda x: np.round(x["Distance"]), axis = 1) 
    sns.scatterplot(x ="rounded_distance", y = "Goal", data = pd.DataFrame(dataset.groupby("rounded_distance").agg("mean")["Goal"]).reset_index())

    plt.title('Distance vs Goal Probability', fontsize = 17, weight = "bold")

    plt.xlabel("Distance", fontsize = 12)
    plt.ylabel("Goal Probability", fontsize = 12)

def plt_angle_goals(dataset): 
    """ 
    Plot angle vs. Goal Probability. 
    
    :param dataset: input shots dataset to graph 
    :type dataset: Pandas DataFrame
    
    :returns: None 
    :type returns: None 
    
    """ 
    
    assert isinstance(dataset, pd.DataFrame)
    assert len(dataset) > 0
    
    plt.figure(figsize = (7, 5))
    dataset["rounded_angle"] = dataset.apply(lambda x: np.round(x["angle_degrees"]), axis = 1) 

    sns.scatterplot(x = "rounded_angle", y = "Goal", data = pd.DataFrame(dataset.groupby("rounded_angle").agg("mean")["Goal"]).reset_index())
    plt.xlabel('Angle', fontsize = 12)
    plt.ylabel('Probability', fontsize = 12)

    plt.title('Angle vs Goal Probability', weight = "bold", fontsize = 17)
    plt.show() 

def setstyle(): 
    """ 
    Set the style for visualization. 
    
    :returns: None 
    :type returns: None 
    
    """ 
    
    assert True 
    
    plt.figure(figsize = (12.5, 10))
    sns.set_style("darkgrid")
    plt.figure(figsize = (12.5, 10))

    print("Style set!") 

def top10_players(dataset): 
    """ 
    Visualize the number of shots, goals and accuracy of top 10 players using bubble plot (scaterplot). 
    
    :param dataset: dataset of shots 
    :type dataset: Pandas DataFrame
    
    :returns: None 
    :type returns: None 
    
    """ 
    
    assert isinstance(dataset, pd.DataFrame)
    assert len(dataset) > 0
    
    top50 = dataset
    
    plt.figure(figsize = (12.5, 10))
    sns.set_style("darkgrid")
    scatterplayers = sns.scatterplot(data = top50[0:10], x = "Goal", y = "Accuracy", size = "Attempts", sizes = (2500, 5000), hue = "shortName", alpha = 0.5, legend = False)

    plt.title("Top 10 Players", weight = "bold", fontsize = 20)

    plt.xlabel("Number of Goals", fontsize = 17)
    plt.ylabel("Accuracy", fontsize = 17)

    for i, label in enumerate(top50[0:10]["shortName"]): 
        plt.text(top50[0:10]["Goal"].iloc[i], top50[0:10]["Accuracy"].iloc[i], label, ha = "center", weight = "semibold") 

def top10_teams(dataset): 
    """ 
    Visualize the number of shots, goals and accuracy of top 10 teams using bubble plot (scaterplot). 
    
    :param dataset: dataset of shots 
    :type dataset: Pandas DataFrame
    
    :returns: None 
    :type returns: None 
    
    """ 
    
    assert isinstance(dataset, pd.DataFrame)
    assert len(dataset) > 0
    
    top50 = dataset

    plt.figure(figsize = (12.5, 10))
    scatterplayers = sns.scatterplot(data = top50[0:10], x = "Goal", y = "Accuracy", size = "Attempts", sizes = (2500, 5000), hue = "name", alpha = 0.5, legend = False)

    plt.title("Top 10 Teams", weight = "bold", fontsize = 20)

    plt.xlabel("Number of Goals", fontsize = 17)
    plt.ylabel("Accuracy", fontsize = 17)

    for i, label in enumerate(top50[0:10]["name"]): 
        plt.text(top50[0:10]["Goal"].iloc[i], top50[0:10]["Accuracy"].iloc[i], label, ha = "center", weight = "semibold")

def plt_location_goals(dataset): 
    """ 
    Plot location vs. Goal Probability. 
    
    :param dataset: input shots dataset to graph 
    :type dataset: Pandas DataFrame
    
    :returns: None 
    :type returns: None 
    
    """ 
    
    assert isinstance(dataset, pd.DataFrame)
    assert len(dataset) > 0
    
    path_eff = [path_effects.Stroke(linewidth = 1.5, foreground = 'black'), path_effects.Normal()]
    pitch = VerticalPitch(pitch_color = '#f4edf0', line_color = 'black',line_zorder = 2, stripe = False, pitch_type = 'wyscout')
    fig, ax = pitch.draw(figsize = (10, 12))

    bin_statistic = pitch.bin_statistic(dataset["X"].astype(float), dataset["Y"].astype(float), statistic = 'count', bins = (20, 15), normalize = True)

    pitch.heatmap(bin_statistic, ax = ax, cmap = 'Reds', edgecolor = '#f9f9f9')
    labels = pitch.label_heatmap(bin_statistic, color = '#f4edf0', fontsize = 12, ax = ax, ha = 'center', va = 'center', str_format = '{:.0%}', path_effects = path_eff, exclude_zeros = True) 

def correlation_heatmap(dataset): 
    """ 
    Plots heatmap with regards to features in dataset. 
    
    :param dataset: dataset to graph 
    :type dataset: Pandas DataFrame 
    
    :returns: None 
    :type returns: None 
    
    """ 
    
    assert isinstance(dataset, pd.DataFrame) 
    assert len(dataset) > 0 
    
    plt.figure(figsize = (12, 7))
    htmap = dataset[["Goal", "Header", "Counter Attack", "strong foot", "Blocked", "First Half", "Distance", "angle_degrees"]]
    htm = sns.heatmap(data = htmap.corr(), annot = True, fmt = ".2f", linewidths = 0.3, linecolor = "purple", cmap = "RdBu", annot_kws = {"weight": "semibold"})

    plt.title("Correlation", fontsize = 17, weight = "bold") 

    htm.set_xticklabels(htm.get_xmajorticklabels(), weight = "semibold")
    htm.set_yticklabels(htm.get_xmajorticklabels(), weight = "semibold") 















