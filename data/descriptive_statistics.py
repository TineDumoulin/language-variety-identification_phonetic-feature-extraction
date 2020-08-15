# Descriptive statistics
from csv import DictReader

VL_uitingen = 0
NL_uitingen = 0

VL_woorden = 0
NL_woorden = 0

session_ids = []
NL_sessions = 0
VL_sessions = 0

with open('processed_data.csv', 'r') as read_obj:
    csv_dict_reader = DictReader(read_obj)
    for row in csv_dict_reader:
        if row['language_classification'] == 'NL':
            NL_uitingen += 1
            NL_woorden += len(row['phonetic_transcription'])
        elif row['language_classification'] == 'VL':
            VL_uitingen += 1
            VL_woorden += len(row['phonetic_transcription'])
        if not row['session_id'] in session_ids:
            session_ids.append(row['session_id'])

for session in session_ids:
    if session.startswith('fn'):
        NL_sessions += 1
    elif session.startswith('fv'):
        VL_sessions += 1

print('VL_uitingen: ', VL_uitingen)
print('VL_woorden: ', VL_woorden)
print('VL_sessions: ', VL_sessions)
print()

print('NL_uitingen: ', NL_uitingen)
print('NL_woorden: ', NL_woorden)
print('NL_sessions: ', NL_sessions)
print()

print('session_ids: ', len(session_ids))


