# Data cleaning
# assimilations, deletions and insertions will be removed
# '[]' indicate a sound (or soundgroup) that was unrecognizable > the word in which this sound occurs will be removed
# '#' indicates a non-linguistic sound > these will be removed altogether

from csv import DictReader

assimilations = ['SEP', 'SHARE-P', 'SHARE-NP', 'SHARE-W', 'INSERT']

counter_hashtag = 0
counter_squarebracket = 0
counter_assimilation = 0

with open('unprocessed_data.csv', 'r') as read_obj:
    csv_dict_reader = DictReader(read_obj)
    for row in csv_dict_reader:
        line = row['phonetic_transcription']
        for word in line.split(' '):
            # Deleting [] words
            if '[]' in word:
                #delete word
                counter_squarebracket += 1
            # Deleting # sounds
            if '#' == word:
                #delete word
                counter_hashtag += 1
        if not any(ele in line for ele in assimilations):
            # keep entry
            continue
        else:
            # delete complete entry
            counter_assimilation += 1

print('counter_hashtag: ', counter_hashtag)
print('counter_squarebracket: ', counter_squarebracket)
print('counter_assimilation: ', counter_assimilation)
