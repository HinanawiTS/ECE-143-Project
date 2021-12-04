import matplotlib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from PlotPitch import draw_pitch
#from sklearn.pipeline import make_pipeline


def logistic_dist(x_train_dis_2, y_train_dis_2):
    """
    Logistic Regression Model between goal and Distance
    """
    assert isinstance(x_train_dis_2, pd.DataFrame)
    assert isinstance(y_train_dis_2, pd.Series)
    poly = PolynomialFeatures(degree = 2, interaction_only=False, include_bias=False)
    lgm_dis_2 = LogisticRegression()
    lgm_dis_2.fit(x_train_dis_2,y_train_dis_2)
    pipe = Pipeline([('polynomial_features',poly), ('logistic_regression',lgm_dis_2)])
    pipe.fit(x_train_dis_2, y_train_dis_2)

    return lgm_dis_2.coef_, lgm_dis_2.intercept_

def plot_prob_goal_dist(df, coef, intercept):
    """
    Plot Logistic Regression Model between goal and distance
    """
    assert isinstance(df, pd.DataFrame)
    assert isinstance(coef, np.ndarray)
    assert isinstance(intercept, np.ndarray)
    fig, axes = plt.subplots(figsize=(11, 5))
    #first we want to create bins to calc our probability
    #pandas has a function qcut that evenly distibutes the data 
    #into n bins based on a desired column value
    df['Goal']=df['Goal'].astype(int)
    df['Distance_Bins'] = pd.qcut(df['Distance'],q=100)
    #now we want to find the mean of the Goal column(our prob density) for each bin
    #and the mean of the distance for each bin
    dist_prob = df.groupby('Distance_Bins',as_index=False)['Goal'].mean()['Goal']
    dist_mean = df.groupby('Distance_Bins',as_index=False)['Distance'].mean()['Distance']
    dist_trend = sns.scatterplot(x=dist_mean,y=dist_prob)
    dist_trend.set_xlabel("Distance (m)", fontsize=12)
    dist_trend.set_ylabel("Probabilty of Goal", fontsize=12)
    dist_trend.set_title("Probability of Scoring Based on Distance", fontsize=17, weight = "bold")
    dis = np.linspace(0,50,100)
    sns.lineplot(x = dis,y = 1/(1+np.exp((coef[0][0]*dis-coef[0][1]*dis**2-intercept[0]))),color='green',
                label='Log Fit with Quadratic Term')
    
    plt.show()

def logistic_angle(X_train, Y_train):
    """
    Logistic Regression Model between goal and angle
    """
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(Y_train, pd.Series)
    poly = PolynomialFeatures(degree = 2, interaction_only=False, include_bias=False)
    lr_ang_poly = LogisticRegression()
    pipe = Pipeline([('polynomial_features',poly), ('logistic_regression',lr_ang_poly)])
    pipe.fit(X_train, Y_train)

    log_odds = lr_ang_poly.coef_[0]

    return lr_ang_poly.coef_, lr_ang_poly.intercept_

def plot_prob_goal_angle(df, coef, intercept):
    """
    Plot Logistic Regression Model between goal and angle
    """
    assert isinstance(df, pd.DataFrame)
    assert isinstance(coef, np.ndarray)
    assert isinstance(intercept, np.ndarray)
    fig, axes = plt.subplots(figsize=(11, 5))
    df['Angle_Bins'] = pd.qcut(df['Angle Degrees'],q=100)
    angle_prob = df.groupby('Angle_Bins',as_index=False)['Goal'].mean()['Goal']
    angle_mean = df.groupby('Angle_Bins',as_index=False)['Angle Degrees'].mean()['Angle Degrees']
    angle_trend = sns.scatterplot(x=angle_mean,y=angle_prob)
    angle_trend.set_xlabel("Avg. Angle of Bin", fontsize=12)
    angle_trend.set_ylabel("Probabilty of Goal", fontsize=12)
    angle_trend.set_title("Probability of Scoring Based on Angle", fontsize=17, weight = "bold")
    ang = np.linspace(0,100,100)
    sns.lineplot(x = ang,y = 1/(1+np.exp(-(coef[0][0]*ang + coef[0][1]*ang**2
                                        + intercept[0]))),color='green', label='Log Fit')

    plt.show()

def logistic_dist_angle(X_train, Y_train):
    """
    Logistic Regression Model 2Dimension
    """
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(Y_train, pd.Series)

    lgm_2 = LogisticRegression(random_state=0)
    lgm_2.fit(X_train,Y_train)
    
    return lgm_2.coef_, lgm_2.intercept_

def Logistic(Y):
    g_y = 1 + np.exp(-Y)
    return np.reciprocal(g_y)

def plot_logistic_model(coef, intercept):
    """
    Plot Logsitc Regression Model
    """
    assert isinstance(coef, np.ndarray)
    assert isinstance(intercept, np.ndarray)
    x0 = np.linspace(-34, 34, 100)
    x1 = np.linspace(.1, 53 , 100)
    x_0 = np.linspace(0, 68, 100)
    x0_grid, x1_grid = np.meshgrid(x0, x1)
    c=7.32
    a=np.sqrt((x0_grid-7.32/2)**2 + x1_grid**2)
    b=np.sqrt((x0_grid+7.32/2)**2 + x1_grid**2)
    h_grid = Logistic(coef[0][1]*np.arccos((c**2-a**2-b**2)/(-2*a*b))
                    +coef[0][0]*np.sqrt((x1_grid)**2+(x0_grid)**2)+intercept[0])


    fig, ax = plt.subplots(figsize=(11, 7))
    draw_pitch(orientation="vertical",
            aspect="half",
            pitch_color='white',
            line_color="black",
            ax=ax)


    CS =plt.contourf(x_0,x1, h_grid,alpha=.85,cmap='OrRd',levels=50)


    plt.title('xG Model', fontsize=17, weight = "bold")

    #plt.axis('off')
    ax.set_xlim(0,68)
    ax.set_ylim(52.5,0)
    plt.colorbar()

    plt.show()

# Define a class that forces representation of float to look a certain way
# This remove trailing zero so '1.0' becomes '1'
class nf(float):
    def __repr__(self):
        str = '%.1f' % (self.__float__(),)
        if str[-1] == '0':
            return '%.0f' % self.__float__()
        else:
            return '%.1f' % self.__float__()

def plot_logistic_contour(coef, intercept):
    """
    Logistic Regression Model 2Dimension Contour
    """
    assert isinstance(coef, np.ndarray)
    assert isinstance(intercept, np.ndarray)
    matplotlib.rcParams['xtick.direction'] = 'out'
    matplotlib.rcParams['ytick.direction'] = 'out'
    
    x0 = np.linspace(-34, 34, 100)
    x1 = np.linspace(.1, 53 , 100)
    x_0 = np.linspace(0, 68, 100)
    x0_grid, x1_grid = np.meshgrid(x0, x1)
    c=7.32
    a=np.sqrt((x0_grid-7.32/2)**2 + x1_grid**2)
    b=np.sqrt((x0_grid+7.32/2)**2 + x1_grid**2)
    h_grid = Logistic(1.57148079*np.arccos((c**2-a**2-b**2)/(-2*a*b))
                    -0.11023242*np.sqrt((x1_grid)**2+(x0_grid)**2)-1.02645936)


    fig, ax = plt.subplots(figsize=(11, 7))
    draw_pitch(orientation="vertical",
            aspect="half",
            pitch_color='white',
            line_color="black",
            ax=ax)


    CS =plt.contour(x_0,x1, h_grid,alpha=1,cmap='OrRd',levels=7)

    # Recast levels to new class
    CS.levels = [nf(val*100) for val in CS.levels]

    # Label levels with specially formatted floats
    if plt.rcParams["text.usetex"]:
        fmt = r'%r \%%'
    else:
        fmt = '%r %%'
    plt.clabel(CS, CS.levels[1::2],inline=True, fmt=fmt, fontsize=12)

    plt.title('xG Model', fontsize=17, weight = "bold")

    #plt.axis('off')
    ax.set_xlim(10,58)
    ax.set_ylim(22,0)

    plt.show()