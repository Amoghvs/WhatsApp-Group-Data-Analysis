# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 21:59:38 2020

@author: Amogh
"""



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import datetime
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud


def plot_bar_chart(total_items, y_values, x_values, title, xlabel, ylabel, rotation=0):
    plt.figure(figsize=(18,10))
    plt.bar(np.arange(total_items), y_values)
    plt.xticks(np.arange(total_items),x_values,rotation=rotation)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def startsWithDateTime(s):
    pattern = '^(\d+/\d+/\d+), ([0-9][0-9]):([0-9][0-9]) -'
    #pattern2='\[[0-9]{1,2}\/[0-9]{1,2}\/[0-9]{2}, [0-9]{1,2}:[0-9]{2}:[0-9]{1,2} [A-Z]{2}] ' 
    result = re.match(pattern, s)
    if result:
        return True
    return False

def startsWithAuthor(s):
    patterns = [
        '([\w]+):',                        # First Name
        '([\w]+[\s]+[\w]+):',              # First Name + Last Name
        '([\w]+[\s]+[\w]+[\s]+[\w]+):',    # First Name + Middle Name + Last Name
        '([+]\d{2} \d{5} \d{5}):',         # Mobile Number (India)
        '([+]\d{2} \d{3} \d{3} \d{4}):',   # Mobile Number (US)
        '([+]\d{2} \d{4} \d{7})'           # Mobile Number (Europe)
    ]
    pattern = '^' + '|'.join(patterns)
    result = re.match(pattern, s)
    if result:
        return True
    return False


def getDataPoint(line):
    # line = 18/06/17, 22:47 - Loki: Why do you have 2 numbers, Banner?
    
    splitLine = line.split(' - ') # splitLine = ['18/06/17, 22:47', 'Loki: Why do you have 2 numbers, Banner?']
    
    dateTime = splitLine[0] # dateTime = '18/06/17, 22:47'
    
    date, time = dateTime.split(', ') # date = '18/06/17'; time = '22:47'
    
    message = ' '.join(splitLine[1:]) # message = 'Loki: Why do you have 2 numbers, Banner?'
    
    if startsWithAuthor(message): # True
        splitMessage = message.split(': ') # splitMessage = ['Loki', 'Why do you have 2 numbers, Banner?']
        author = splitMessage[0] # author = 'Loki'
        message = ' '.join(splitMessage[1:]) # message = 'Why do you have 2 numbers, Banner?'
    else:
        author = None
    return date, time, author, message

parsedData = [] # List to keep track of data so it can be used by a Pandas dataframe



with open('whatsapp-chat.txt', encoding="utf-8") as fp:
    fp.readline() # Skipping first line of the file (usually contains information about end-to-end encryption)
        
    messageBuffer = [] # Buffer to capture intermediate output for multi-line messages
    date, time, author = None, None, None # Intermediate variables to keep track of the current message being processed
    
    while True:
        line = fp.readline() 
        if not line: # Stop reading further if end of file has been reached
            break
        line = line.strip() # Guarding against erroneous leading and trailing whitespaces
        if startsWithDateTime(line): # If a line starts with a Date Time pattern, then this indicates the beginning of a new message
            if len(messageBuffer) > 0: # Check if the message buffer contains characters from previous iterations
                parsedData.append([date, time, author, ' '.join(messageBuffer)]) # Save the tokens from the previous message in parsedData
            messageBuffer.clear() # Clear the message buffer so that it can be used for the next message
            date, time, author, message = getDataPoint(line) # Identify and extract tokens from the line
            messageBuffer.append(message) # Append message to buffer
        else:
            messageBuffer.append(line) # If a line doesn't start with a Date Time pattern, then it is part of a multi-line message. So, just append to buffer




df = pd.DataFrame(parsedData, columns=['Date', 'Time', 'Author', 'Message'])
df.head()


"""Names of Members"""
df['Author'].unique()


"""Top 10 Active People"""
author_value_counts = df['Author'].value_counts() # Number of messages per author
top_10_author_value_counts = author_value_counts.head(10) # Number of messages per author for the top 10 most active authors
top_10_author_value_counts.plot.bar() # Plot a bar chart using pandas built-in plotting apis
plt.ylabel('Number of msgs')
plt.title('Top 10 active people')


media_messages_df = df[df['Message'] == '<Media omitted>']
print(media_messages_df.head())

"""Top 10 People with most media sent"""
author_media_messages_value_counts = media_messages_df['Author'].value_counts()
top_10_author_media_messages_value_counts = author_media_messages_value_counts.head(10)
top_10_author_media_messages_value_counts.plot.bar()
plt.ylabel('Media Msgs count')
plt.title('Most media sent')

null_authors_df = df[df['Author'].isnull()]
null_authors_df.head()


messages_df = df.drop(null_authors_df.index) # Drops all rows of the data frame containing messages from null authors
messages_df = messages_df.drop(media_messages_df.index) # Drops all rows of the data frame containing media messages
messages_df.head()

messages_df['Letter_Count'] = messages_df['Message'].apply(lambda s : len(s))
messages_df['Word_Count'] = messages_df['Message'].apply(lambda s : len(s.split(' ')))

#concating into Date Time
df["DateTime"] = df["Date"] +" "+ df["Time"]
#converting into Datetime timeseries
df['DateTime']=pd.to_datetime(df['DateTime'])

df=df.drop(0)
df.head()

"""Messages Traffic"""

messages_df['Hour'] = messages_df['Time'].apply(lambda x : x.split(':')[0]) # The first token of a value in the Time Column contains the hour (Eg., "20" in "20:15")
messages_df['Hour'].value_counts().head(24).sort_index(ascending=False).plot.bar() # Top 10 Hours of the day during which the most number of messages were sent
plt.ylabel('Number of messages')
plt.xlabel('Hour of Day')
plt.title('Messages Traffic')






""""Deleted Message"""
deleted_messages_df = df[df['Message'] == 'This message was deleted']
print(deleted_messages_df.head())

author_deleted_messages_value_counts = deleted_messages_df['Author'].value_counts()
top_author_deleted_messages_value_counts = author_deleted_messages_value_counts.head(10)
top_author_deleted_messages_value_counts.plot.bar()
plt.ylabel('Deleted Msgs count')
plt.title('Deleted Msgs')





from datetime import datetime
import time
start = datetime.now()
end = df.iloc[1]['DateTime']
print(start)
print(end)
tot_days=start - end
tot_days
print("Chats Since", tot_days)


"""Time Series"""

import plotly.express as px


# Plotd
count=df.Message.count()
plt.plot(df['DateTime'], count)

df.groupby(['Time']).count()['Message'].plot()

df.groupby(['Date']).count()['Message'].plot()

df.groupby(['DateTime']).count()['Message'].plot()

import matplotlib.pyplot as plt
plt.style.use('ggplot')

date=df.groupby('Date')['Date'].count().nlargest(10)
datetime=df.groupby('DateTime')['DateTime'].count().nlargest(10)
def user_line_chart(date):
    ax = date.plot(kind='line',color='green', fontsize=12)
    ax.set_title("Distribution of Date with User Chats\n", fontsize=18)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Chats", fontsize=12)
    plt.show()

user_line_chart(date)

"""Word cloud"""

df['Author'].unique()

import matplotlib.pyplot as pPlot
from wordcloud import WordCloud, STOPWORDS
import numpy as npy
from PIL import Image



#image_mask = np.array(Image.open("professor_snape.png"))
text = ' '.join(df['Message'])

import nltk
words = set(nltk.corpus.words.words())

#sent = "Io andiamo to the beach with my amico."
" ".join(w for w in nltk.wordpunct_tokenize(text) \
         if w.lower() in words or not w.isalpha())


#wordcloud2
wordcloud = WordCloud(width=1600, height=800).generate(text)
# Open a plot of the generated image.

plt.figure( figsize=(20,10), facecolor='k')
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()