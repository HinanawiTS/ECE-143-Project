# Expected Goals Analysis 
An analysis of soccer goals, focusing on discovering what factors affect the probability of getting a goal, and constructing an Expected Goals (xG) model based on features we selected during the data analysis process. 


## Dataset

[Wyscout Events Dataset](https://figshare.com/collections/Soccer_match_event_dataset/4415000/2): A relatively new, large and clean dataset containing all notable actions recorded in a season of professional European soccer games, such as passes, fouls, shots, etc, and their corresponding metadata. Last updated in early 2020 (with new features).  All available datasets are required. 


## Repository Structure 
    .
    ├── data  
        ├── players.json
        ├── teams.json
        ├── other .json and .csv files, except events
        ├── matches
            ├── all matches .json files(7 files)
        ├── events
            ├── all events .json files (7 files) 
            
    ├── src 
        ├── cleaning_and_eda.py # cleaning and EDA 
        ├── visualization.py    # Data Analysis and Generating Corresponding Visualizations 
        ├── prediction_analysis.py         # Training and prediction with 9 different models
        ├── shot_martrix.py # Get a tiddy dataset about goals, distance and angle
        ├── PlotPitch.py # Plot football filed more visually
        ├── logistic_plus.py # Plot Logistic Regression 
        
    ├── Project.ipynb           # Project Notebook 
    ├── Project Report.pdf      # Project Report 
    ├── requirements.txt        # Requirements 
    ├── .gitignore              
    ├── LICENSE
    └── README.md

## Requirements 
    os
    sys
    tqdm.notebook
    pathlib
    urllib.parse
    urllib.request
    zipfile 
    numpy 
    json 
    pandas 
    seaborn 
    mplsoccer
    matplotlib 
    sklearn
    glob
    scipy

## How to Run the Project
Install all necessary packages.
  
Run the Project Notebook in the root repository.   





