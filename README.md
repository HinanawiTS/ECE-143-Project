# Expected Goals Analysis 
An analysis of soccer goals, focusing on discovering what factors affect the probability of getting a goal, and constructing an Expected Goals (xG) model based on features we selected during the data analysis process. 


## Dataset

[Wyscout Events Dataset](https://figshare.com/collections/Soccer_match_event_dataset/4415000/2): A relatively new, large and clean dataset containing all notable actions recorded in a season of professional European soccer games, such as passes, fouls, shots, etc, and their corresponding metadata. All available datasets are required. 


## Repository Structure 
    .
    ├── data  
        ├── players.json
        ├── teams.json
        ├── other .json and .csv files, except events 
        ├── events
            ├── all events .json files (7 files) 
            
    ├── src 
        ├── cleaning_and_eda.py # cleaning and EDA 
        ├── visualization.py    # Data Analysis and Generating Corresponding Visualizations 
        
    ├── Project.ipynb           # Project Notebook
    ├── requirements.txt        
    ├── .gitignore              
    ├── LICENSE
    └── README.md

## Requirements 
    os 
    numpy 
    json 
    pandas 
    seaborn 
    mplsoccer 
    sys 
    matplotlib 
    sklearn
    scipy  





## How to Run the Project
Install all necessary packages and download all required [datasets](https://figshare.com/collections/Soccer_match_event_dataset/4415000/2). 



Put datasets into /data and /data/events based on the structure mentioned above. 
    
    
Run the Project Notebook in the root repository.  
