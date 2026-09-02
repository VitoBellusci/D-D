import requests
import json
import os
import pathlib
import io
from PIL import Image
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from DndDataset import myData, MyCollate
from Vocabulary import Vocabulary

url = 'https://www.dnd5eapi.co/api/2014/monsters'
payload = requests.get(url)

# validazione risposta
payload.raise_for_status()

monsters = payload.json()

collection = monsters['results']

captions_dataset = []

for monster in collection:
    monster_url = 'https://www.dnd5eapi.co' + monster['url']
    response = requests.get(monster_url)
    response.raise_for_status()
    details = response.json()

    armor_class_list = details.get('armor_class', [])
    armor_class = [str(item.get('value')) for item in armor_class_list if 'value' in item]

    actions_list = details.get('actions', [])
    actions = [f"{item.get('name', '')}, {item.get('desc', '')}" for item in actions_list]


    raw_traits = details.get('special_abilities', [])
    traits = [f"{item.get('name', '')}, {item.get('desc', '')}" for item in raw_traits]

    damage_resistances = ', '.join(details.get('damage_resistances', [])) or "none"

    condition_immunities_list = details.get('condition_immunities', [])
    condition_immunities = ', '.join([str(item.get('name', '')) for item in condition_immunities_list]) or "none"

    senses = ', '.join(f"{k.replace('_', ' ')}: {v}" for k, v in details.get('senses', {}).items())
    languages = details.get('languages', '') or "none"

    proficiencies = details.get('proficiencies', [])
    saving_throws = ', '.join([
        p['proficiency']['name'].replace('Saving Throw: ', '') 
        for p in proficiencies if 'Saving Throw' in p.get('proficiency', {}).get('name', '')
    ]) or "none"

    stats = {
        'name': details.get('name', ''),
        'type': details.get('type', ''),
        'armor_class': ', '.join(armor_class),
        'HP': details.get('hit_points', ''),
        'speed': ', '.join(f"{k}: {v}" for k, v in details.get('speed', {}).items()),
        'damage_resistances': damage_resistances,
        'condition_immunities': condition_immunities,
        'senses': senses,
        'actions': ', '.join(actions),
        'traits': ', '.join(traits),
        'challenge_rating': details.get('challenge_rating', ''),
        'languages': languages,
        'strength': details.get('strength', ''),
        'dexterity': details.get('dexterity', ''),
        'constitution': details.get('constitution', ''),
        'intelligence': details.get('intelligence', ''),
        'wisdom': details.get('wisdom', ''),
        'charisma': details.get('charisma', ''),
        'saving_throws': saving_throws
    }

    caption = (
        f"The {stats['name']} is a {stats['type']}. "
        f"It has an Armor Class of {stats['armor_class']} and {stats['HP']} Hit Points. "
        f"Its speed is {stats['speed']}. "
        f"Its characteristics are STR {stats['strength']}, DEX {stats['dexterity']}, CON {stats['constitution']}, "
        f"INT {stats['intelligence']}, WIS {stats['wisdom']}, and CHA {stats['charisma']}. "
        f"It has saving throw proficiencies in {stats['saving_throws']}. "
        f"Damage resistances include {stats['damage_resistances']}, and condition immunities include {stats['condition_immunities']}. "
        f"Its senses are {stats['senses']} and it speaks {stats['languages']}. "
        f"It has a Challenge Rating of {stats['challenge_rating']}. "
        f"Special traits include: {stats['traits']}. "
        f"Actions available: {stats['actions']}."
    )

    index = stats['name']
    image_path = f"./assets/{monster['index']}.jpg"

    if os.path.exists(image_path):
        m = {'image': image_path, 'caption': caption}
        captions_dataset.append(m)
    else:
        print('Immagine non trovata')
        continue

df = pd.DataFrame(captions_dataset)
df.to_csv('monsters_dataset.csv', index=False)

train_df, test_df = train_test_split(df, test_size=0.15, train_size=0.85, random_state=123, shuffle=True)

train_df, val_df = train_test_split(train_df, test_size=0.18, random_state=123, shuffle=True)


train_df.to_csv('monsters_train.csv', index=False)
test_df.to_csv('monsters_test.csv', index=False)
val_df.to_csv('monsters_val.csv', index=False)