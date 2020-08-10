import os


# Stap 1: Get all files > list van alle file paths!
files = open(r"C:\Users\tine-\OneDrive\Documents_bak\School\Ma_Taalkunde\Masterproef\data\filenames.txt").readlines()
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

paths = []
for file in files:
    path = find_file(file, r"C:\COREX6\data\annot\corex\sea")
    paths.append(path)


# Stap 2: Get orth-list and pho-list van elke file > OPGEPAST: PHO en ORT nog verwijderen van begin van de lijn!
def get_lines(file_name, string_to_search):
    """ Check if any line in the file contains given string """
    # Open the file in read only mode
    list = []
    with open(file_name, 'r') as read_obj:
        # Read all lines in the file one by one
        for line in read_obj:
            # For each line, check if line contains the string
            if string_to_search in line:
                line = line.replace(string_to_search, '').strip()
                list.append(line)
    return list

# Stap 3: Map orth-list op alle mogelijke pho's van elke file
# Check if word already in mapping, otherwise create object
ort_pho_mappings = {}

for file in paths:
    ort_lines = get_lines(file, 'ORT ')
    pho_lines = get_lines(file, 'PHO ')

    for (line_idx, line) in enumerate(ort_lines):
        words_ort = line.split()
        words_pho = pho_lines[line_idx].split()

        for (word_idx, word) in enumerate(words_ort):
            if word not in ort_pho_mappings:
                # add object with empty pronunciation lists
                ort_pho_mappings[word] = {"NL": [], "VL": []}
            #print(ort_pho_mappings[word])

            region_mapping = "NL" if os.path.basename(file).startswith('fn') else "VL"

            # Check if pronunciation already in mapping,
            try:
                current_pho = words_pho[word_idx]
                region_list = ort_pho_mappings[word][region_mapping]

                if current_pho not in region_list:
                    region_list.append(current_pho)
            except:
                pass

import json
with open('data.json', 'w') as fp:
    json.dump(ort_pho_mappings, fp, indent=2)