"""
10 supporting functions for the analysis in the jupyter notebook file:
Function to extract information from several datafiles, to print descriptives, and to plot graphs for analysis.
"""

import pandas as pd
import re
from wordcloud import WordCloud
from nltk.corpus import stopwords
import emoji
from collections import Counter
import matplotlib.pyplot as plt, numpy as np
import config as cfg


def extract_msg(*filenames):

    """
    Transforming one or more raw .txt whatsapp-file/s by reading chat file and seperating date, message and sender.
    :param filenames: list all filenames you want to add to the analysis
    :return: .pd dataframe with columns: date, msg, sender
    """
    df_combined = pd.DataFrame()
    for filename in filenames:
        data = pd.read_fwf(filename, sep=" ", header=None, encoding='utf-8')
        date = []
        msg = []
        sender = []
        datetime_pat = "\d{2}.\d{2}.\d{2}\,\ \d{2}:\d{2}:\d{2}"
        sender_pat = '] (.*?):'
        for line in data[0]:
            try:
                d = (re.search(datetime_pat, line)).group(0)
                date.append(d)
            except:
                date.append('')
            try:
                s = (re.search(sender_pat, line)).group(1)
                sender.append(s)
            except:
                sender.append(np.nan)
            try:
                msg.append(line.split(': ', 1)[1])
            except:
                msg.append(line)
        df = pd.DataFrame(list(zip(date, sender, msg)), columns=['date', 'sender', 'message'])
        df = df[~df.message.isnull()]
        df = df[~df.message.str.contains(datetime_pat)] #not including actions without date, such as 'changed icon', 'added person'
        df = df[~df.message.str.contains('Messages to this group are now secured with end-to-end encryption')] #deleted chatname-actions
        df.date = pd.to_datetime(df.date)
        df.date.ffill(axis=0, inplace = True)
        df.sender.ffill(axis=0, inplace = True) # added dates to consecutive messages
        df_combined = df_combined.append(df, ignore_index=True) #combining manipulated dfs from different chats
        # replace sender names
        list_members_og = cfg.family_members_og # list of family
        list_members_new_names = cfg.family_members # list of og_names in chat list(df_combined.sender.unique())
        dict_senders = dict(zip(list_members_og, list_members_new_names))
        df_combined.replace(dict_senders, inplace=True)
        df_combined.sort_values(by='date', inplace=True, ascending=False)
    return df_combined

#***********************************************
def describe_msgs(df):
    print('\nThe first message was sent on: ', df.date.min())
    print('\nThe last message was sent', round((((df.date.max() - df.date.min()).days)/365), 2), 'years later, on: ', df.date.max())
    print('\nNumber of messages sent', len(df.message))
    print('\nMembers of this chat: ', ', '.join(df.sender.unique()))
    print('\nMost messages were sent by:', df['sender'].value_counts().idxmax())


def monologue(df):
    """
    Identifying the sender who sent the most messages without a response
    """
    prev_sender = []
    max_spam = 0
    num_spam = 0
    for i in range(len(df)):
        current_sender = df['sender'].iloc[i]
        if current_sender == prev_sender:
            num_spam += 1
            if num_spam > max_spam:
                max_spam = num_spam
                max_spammer = current_sender
        else:
            num_spam = 0
        prev_sender = current_sender
    print("The longest monologue is from %s with %d consecutive lines of messages" % (max_spammer, max_spam))

#***********************************************

def plot_msg_pp(df):
    """
    Barplot: messages per person
    """
    counts_perc = df['sender'].value_counts() / len(df) * 100  # in percent
    counts_perc.plot(kind="bar", title="% Messages per person")


def plot_most_media(df):
    """
    Barplot: most images/videos/gifs sent
    """
    gifs_sent = {}
    for sender in df['sender'].unique():
        gifs_sent[sender] = 0
    for jj in range(len(df)):
        if "omitted" in df["message"].iloc[jj]:
            gifs_sent[df['sender'].iloc[jj]] += 1
    gifs_pd = pd.DataFrame.from_dict(gifs_sent, orient="index")
    gifs_pd.sort_values(by=0, ascending=False, inplace=True)
    gifs_pd = gifs_pd.transpose().iloc[0]
    names = list(df.sender.unique())
    _ = gifs_pd.plot(kind='bar', legend=False, title="Most images/videos/gifs sent", color = "green")


def plot_most_haha(df):
    """
    Barplot: most "haha" sent
    """
    haha_sent = {}
    for sender in df['sender'].unique():
        haha_sent[sender] = 0
    haha = ["ha","haha","hehe","hihi","HAHA", "HAHAHA", "hahaha"]
    for jj in range(len(df)):
        if any(x in df["message"].iloc[jj].lower() for x in haha):
            haha_sent[df['sender'].iloc[jj]] += 1
    haha_pd  = pd.DataFrame.from_dict(haha_sent,orient="index")
    haha_pd.sort_values(by=0,ascending=False, inplace=True)
    haha_pd = haha_pd.transpose().iloc[0]
    _ =haha_pd.plot(kind='bar', legend = False, title = "Most Haha", color="yellow")



def plot_most_YELLING(df):
    """
    Barplot: Most capitalized words sent
    """
    yelling_sent = {}
    for sender in df['sender'].unique():
        yelling_sent[sender] = 0
    for jj in range(len(df)):
        if df["message"].iloc[jj].upper() == df["message"].iloc[jj]:
            yelling_sent[df['sender'].iloc[jj]] += 1
    yelling_pd  = pd.DataFrame.from_dict(yelling_sent,orient="index")
    yelling_pd.sort_values(by=0,ascending=False, inplace=True)
    yelling_pd = yelling_pd.transpose().iloc[0]
    _ = yelling_pd.plot(kind='bar', legend = False, title = "MOST YELLING", color="orange")


def plot_wordcloud(df, language = 'german', max_words=100):
    """
    Combining all text to one string to then create a wordcloud
    """
    text=df.message.str.cat(sep=', ')
    stoplist = ['omitted', 'image', "video"]
    words = stopwords.words(language)
    stoplist = stoplist + words
    wordcloud = WordCloud(font_path='/Library/Fonts/arial.ttf', stopwords = stoplist, max_font_size=50, max_words=max_words, background_color="white").generate(text)
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()



def plot_emojis(df):
    """
    Barplot: most emoji sent
    """
    text = df.message.str.cat(sep=', ')
    emj = []
    for e in text:
        if e in emoji.UNICODE_EMOJI:
            emj.append(e)
    grouped = Counter(emj)
    counted = dict(grouped)
    e_df = pd.DataFrame.from_dict(counted, orient='index', columns=['count'])
    e_df.sort_values(ascending=False, by=['count'],inplace=True)
    most_e = e_df.head(10)

    freqs = list(most_e['count'])
    labels = list(most_e.index)
    plt.figure(figsize=(12, 8))
    p1 = plt.bar(np.arange(len(labels)), freqs, 0.8, color="lightblue")
    plt.ylim(0, plt.ylim()[1] + 30)

    # Make labels
    for rect1, label in zip(p1, labels):
        height = rect1.get_height()
        plt.annotate(label,(rect1.get_x() + rect1.get_width() / 2, height + 5), ha="center", va="bottom", fontsize=30)
    plt.show()


def plot_msgs_over_time(df):
    """
    time-series plot: Number of messages sent over time
    """
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    msg = df.groupby(pd.Grouper(freq='2M')).count()
    #plot year v total checkins
    plt.plot(msg.sender)
    plt.xticks(rotation=45)
    plt.show()

