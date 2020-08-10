''' In dit script willen we een csv opstellen in het volgende format:
session_id, fonetic transcription, NL/VL classification'''

'''
# Stap 1: Get all files > lijst van alle file paths!
files = open(r'data\filenames.txt').readlines()
clean_files = []

for file in files:
    if file.endswith('\n'):
        file = file.replace('\n', '')
        clean_files.append(file)

files = clean_files

def find_file(filename, search_path):
   for root, dir, files in os.walk(search_path):
      if filename in files:
         return os.path.join(root, filename)

outF = open("filepaths.txt", "w")

for file in files:
  path = find_file(file, r"C:\COREX6\data\annot\corex\sea")
  outF.write(path)
  outF.write("\n")
outF.close()'''

import os
import csv

# Step 2: Find session_id (a), phonetic transcription (b) en language classification (c)
# !!! remove 'PHO ' @ start of line!

def get_lines(file_name, string_to_search):
    """ Check if any line in the file contains given string. 
    Returns the complete line."""
    lines = []
    with open(file_name, 'r') as read_obj:
        for line in read_obj:
            if line.startswith(string_to_search):
                line = line.replace(string_to_search, '').strip()
                lines.append(line)
        return lines

paths = []
with open("filepaths.txt", "r") as f:
  for line in f:
    stripped_line = line.strip()
    paths.append(stripped_line)

with open('unprocessed_data.csv', 'w', newline='') as csvfile:
    for path in paths:
        results = get_lines(path, 'PHO ')
        for result in results:
            # get session_id
            filename = os.path.basename(path)
            (session_id, extension) = os.path.splitext(filename)

            # get phonetic transcription
            phonetic_transcript = result

            #get language classification
            if session_id.startswith('fn'):
                lang_class = 'NL'
            elif session_id.startswith('fv'):
                lang_class = 'VL'

            # write to CSV
            datawriter = csv.writer(csvfile, quoting=csv.QUOTE_NONNUMERIC, delimiter=';')
            datawriter.writerow([session_id, phonetic_transcript, lang_class])