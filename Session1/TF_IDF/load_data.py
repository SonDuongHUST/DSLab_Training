import re
from os import listdir
from os.path import isfile
from nltk.stem.porter import PorterStemmer

def collect_data_from(parent_dir, newsgroup_list):
    data = []
    for group_id, newsgroup in enumerate(newsgroup_list):
        label = group_id
        dir_path = parent_dir + '/' + newsgroup + '/'
        files = [(filename, dir_path + filename)
                 for filename in listdir(dir_path)
                 if isfile(dir_path + filename)]
        files.sort()
        for filename, filepath in files:
            with open(filepath) as f:
                text = f.read().lower()
                # remove stop words then stem remaining words
                words = [stemmer.stem(word)
                         for word in re.split('\W+', text)
                         if word not in stop_words]
                # combine remaining words
                content = ' '.join(words)
                assert len(content.splitlines()) == 1
                data.append(str(label) + '<fff>' + filename + '<fff>' + content)
    return data


path = '20news-bydate'
dirs = [path + '/' + dir_name + '/'
        for dir_name in listdir(path)
        if not isfile(path + '/' + dir_name)]
train_dir, test_dir = (dirs[0], dirs[1]) if 'train' in dirs[0] \
    else (dirs[1], dirs[0])
list_newsgroups = [newsgroup
                   for newsgroup in listdir(train_dir)]
list_newsgroups.sort()

with open('20news-bydate/stop_words.txt') as f:
    stop_words = f.read().splitlines()
stemmer = PorterStemmer()

train_data = collect_data_from(
    parent_dir=train_dir,
    newsgroup_list=list_newsgroups
)
test_data = collect_data_from(
    parent_dir=test_dir,
    newsgroup_list=list_newsgroups
)

full_data = train_data + test_data
with open('20news-bydate/20news-train-processed.txt', 'w') as f:
    f.write('\n'.join(train_data))
with open('20news-bydate/20news-test-processed.txt', 'w') as f:
    f.write('\n'.join(test_data))
with open('20news-bydate/20news-full-processed.txt', 'w') as f:
    f.write('\n'.join(full_data))

