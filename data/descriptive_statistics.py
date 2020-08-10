# Descriptive statistics
from csv import DictReader

VL_uitingen = 0
NL_uitingen = 0

VL_woorden = 0
NL_woorden = 0

with open('data.csv', 'r') as read_obj:
    csv_dict_reader = DictReader(read_obj)
    for row in csv_dict_reader:
        if row['language_classification'] == 'NL':
            NL_uitingen += 1
            NL_woorden += len(row['phonetic_transcription'])
        elif row['language_classification'] == 'VL':
            VL_uitingen += 1
            VL_woorden += len(row['phonetic_transcription'])

print('VL_uitingen: ', VL_uitingen)
print('VL_woorden: ', VL_woorden)

print('NL_uitingen: ', NL_uitingen)
print('NL_woorden: ', NL_woorden)
