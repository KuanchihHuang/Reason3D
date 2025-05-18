import glob 
import os
from pathlib import Path
import json

symbol_to_room_type = {
    'a': 'bathroom',
    'b': 'bedroom',
    'c': 'closet',
    'd': 'dining room',
    'e': 'entryway',
    'f': 'familyroom',
    'g': 'garage',
    'h': 'hallway',
    'i': 'library',
    'j': 'laundryroom',
    'k': 'kitchen',
    'l': 'living room',
    'm': 'conference room',
    'n': 'lounge',
    'o': 'office',
    'p': 'porch',
    'r': 'game',
    's': 'stairs',
    't': 'toilet',
    'u': 'utility room',
    'v': 'tv',
    'w': 'workout',
    'x': 'outdoor areas',
    'y': 'balcony',
    'z': 'other room',
    'B': 'bar',
    'C': 'classroom',
    'D': 'dining booth',
    'S': 'spa',
    'Z': 'junk',
    #'-' = no label,
}

house_paths = glob.glob("./house_type/*")

region_to_room_type = {}

for house_path in house_paths:

    with open(house_path, 'r') as f:
        room_type = f.readlines()

    house_id = Path(house_path).stem

    for line in room_type:
        if line[0] == 'R':
            region = line.split()[1]
            room = line.split()[5]
            region_to_room_type[house_id+'_region'+region] = symbol_to_room_type[room]

with open('mp3d_room_type.json', 'w') as f:
    json.dump(region_to_room_type, f)


