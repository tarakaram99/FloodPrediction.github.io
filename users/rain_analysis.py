import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as py
import plotly.offline




from django.conf import settings
import os
def analysis():
    filepath = settings.MEDIA_ROOT + "\\" + "rainfall in india 1901-2015.csv"
    df = pd.read_csv(filepath)
    df = df.rename(columns={'SUBDIVISION': 'STATE'})

    #Annual Rainfall received by each State over years
    plt.figure(figsize=(15, 10))
    sns.lineplot(x='YEAR', y='ANNUAL', hue='STATE', data=df)
    plt.title('Annual Rainfall received', fontsize=20)

    # Highest Rainfall ever received in a year in States
    plt.figure(figsize=(15, 8))
    df.groupby(['STATE', 'YEAR'])['ANNUAL'].sum().sort_values(ascending=False).plot()
    plt.grid()
    plt.xlabel("State,Year", fontsize=15)
    plt.ylabel("Annual Rainfall received", fontsize=15)
    plt.title('Highest Rainfall year Data of States', fontsize=20)

    #Total amount of rainfall recieved overall by each state
    plt.figure(figsize=(15, 10))
    df.groupby(['STATE'])['ANNUAL'].sum().sort_values(ascending=False).head(15).plot(kind='bar', color='black')
    plt.ylabel('Total Rainfall')
    plt.title('Total Rainfall Data', fontsize=20)
    plt.grid()

    #The month with the hightest rainfall

    plt.figure(figsize=(10, 7))
    df[['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG',
        'SEP', 'OCT', 'NOV', 'DEC']].mean().plot(kind='bar', color='red')
    plt.xlabel('Months', fontsize=15)
    plt.ylabel('Avg. Rainfall', fontsize=15)
    plt.title('Avg. Monthly Rainfall Data', fontsize=25)
    plt.grid()
    plt.show()

    #Season wise rainfall in India...

    plt.figure(figsize=(10, 2))
    df[['STATE', 'Jan-Feb', 'Mar-May',
        'Jun-Sep', 'Oct-Dec']].groupby("STATE").mean().sort_values('Jun-Sep').plot.bar(width=0.5, edgecolor='k',
                                                                                       align='center', stacked=True,
                                                                                       figsize=(16, 8))
    plt.xlabel('STATE', fontsize=15)
    plt.ylabel('Rainfall (in mm)', fontsize=15)
    plt.title('Rainfall in States of India', fontsize=25)
    plt.grid()

    # Visualizing annual rainfall over the years(1901-2015) in India
    df.groupby("YEAR").mean()['ANNUAL'].plot(ylim=(1000, 2000), color='k', marker='o', markerfacecolor='red',
                                             linestyle='-', linewidth=2, figsize=(12, 10))
    plt.xlabel('Year', fontsize=20)
    plt.ylabel('Annual Rainfall (in mm)', fontsize=20)
    plt.title('Annual Rainfall from Year 1901 to 2015 in India', fontsize=25)
    plt.grid()

    return 'user_home.html'

