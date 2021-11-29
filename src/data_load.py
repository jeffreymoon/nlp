# %%
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline

# %%
DATA_IN_PATH = "../data"

train_data = pd.read_csv(DATA_IN_PATH+"labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)

# print(train_data.head())
print("file size : ")
for file in os.listdir(DATA_IN_PATH):
    if 'tsv' in file and 'zip' not in file:
        print(file.ljust(30) + str(round(os.path.getsize(DATA_IN_PATH + file) / 1000000, 2)) + 'MB')

print(f'Total count of data: {len(train_data)}')

train_length = train_data['review'].apply(len)
print(train_length.head())

# %%
plt.figure(figsize=(12, 5))

plt.hist(train_length, bins=200, alpha=0.5, color='r', label='word')
plt.yscale('log', nonpositive='clip')

plt.title('Log-Histogram of length of review')

plt.xlabel('Length of review')
plt.ylabel('Number of review')

# %%

print(f'Max length of reivew : {np.max(train_length)}')
print(f'Min length of reivew : {np.min(train_length)}')
print(f'Mean length of reivew : {np.mean(train_length)}')
print(f'std length of reivew : {np.std(train_length)}')
print(f'Median length of reivew : {np.median(train_length)}')
print(f'1/4 length of reivew : {np.percentile(train_length, 25)}')
print(f'3/4 length of reivew : {np.percentile(train_length, 75)}')

# %%
plt.figure(figsize=(12, 5))

plt.boxplot(train_length, labels=['counts'], showmeans=True)

# %%
from wordcloud import WordCloud
cloud = WordCloud(width=800, height=600).generate(" ".join(train_data['review']))
plt.figure(figsize=(20, 15))
plt.imshow(cloud)
plt.axis('off')

# %%
fig, axe = plt.subplots(ncols=1)
fig.set_size_inches(6,3)
sns.countplot(train_data['sentiment'])

# %%
print(f"positive review : {train_data['sentiment'].value_counts()[1]}")
print(f"negative review : {train_data['sentiment'].value_counts()[0]}")

# %%
train_word_counts = train_data['review'].apply(lambda x:len(x.split(' ')))

plt.figure(figsize=(15, 10))
plt.hist(train_word_counts, bins=50, facecolor='r', label='train')
plt.title('Log-Histogram of word count in review', fontsize=15)
plt.yscale('log', nonpositive='clip')
plt.legend()
plt.xlabel('Number of words', fontsize=15)
plt.ylabel('Number of words', fontsize=15)

# %%
qmarks = np.mean(train_data['review'].apply(lambda x: '?' in x))
fullstop = np.mean(train_data['review'].apply(lambda x: '.' in x))
capital_first = np.mean(train_data['review'].apply(lambda x: x[0].isupper()))
capitals = np.mean(train_data['review'].apply(lambda x: max([y.isupper() for y in x])))
numbers = np.mean(train_data['review'].apply(lambda x: max([y.isdigit() for y in x])))

print(f'qmarks : {qmarks * 100}')
print(f'fullstop : {fullstop * 100}')
print(f'capital_first : {capital_first * 100}')
print(f'capitals : {capitals * 100}')
print(f'numbers : {numbers * 100}')
