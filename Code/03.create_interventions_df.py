import pandas as pd
import os
import shutil
import ast 
import requests
from bs4 import BeautifulSoup as bs
import re
from rapidfuzz import process, fuzz
import itertools

def get_names(url_lst):
    names = []
    for url in url_lst:
        soup = bs(requests.get(url).content, 'html.parser')
        cat_groups = soup.find_all('div', {'class': 'mw-category-group'})
        for letter in cat_groups:
            people = letter.find_all('li')
            names += [person.text for person in people]
    return list(set(names))

def clean_list(names):
    new_lst = []
    for name in names:
        if name.lower().startswith('anexo'):
            continue
        name = re.sub(r'\([^)]*\)', '', name).lower().strip()
        new_lst.append(name)
    return new_lst

def match_name(text, congress_names):
    """
    Extracts possible names from speaker_text and matches them 
    to the most likely congressperson name.
    """
    congress_names_normalized = {name: name for name in congress_names}

    match = re.search(r'(?:senador|senadora|representante|honorable)\s+(.*)', text, re.IGNORECASE)
    if not match:
        return False

    extracted_tokens = match.group(1).split()
    candidates = [" ".join(extracted_tokens[i:j]) for i in range(len(extracted_tokens)) for j in range(i+1, len(extracted_tokens)+1)]
    candidates += [" ".join(p) for subset in candidates if len(subset.split()) <= 4 for p in itertools.permutations(subset.split())]

    best_match, best_score = None, 0
    for candidate in candidates:
        match, score, _ = process.extractOne(candidate, congress_names_normalized.keys(), scorer=fuzz.token_sort_ratio)
        if score > best_score:
            best_match, best_score = match, score

    return congress_names_normalized[best_match] if best_match and best_score > 70 else False

#chunk the main sessions dataframe
sessions_path = r"../outputs/sessions.csv"
sessions_df = pd.read_csv(sessions_path)

chunk_size = sessions_df.shape[0] // 10
chunks = [sessions_df[i:i + chunk_size] for i in range(0, sessions_df.shape[0], chunk_size)]

# create intervention for chunk
chunk_dir = r"../outputs/temp_chunks"
os.mkdir(chunk_dir)

intervention_id = 0
for chunk in chunks:
    chunk_ints = pd.DataFrame(columns=["session_id", "intervention_id", "speaker_text", "intervention_text"])
    for i, row in chunk.iterrows():
        session_id = row['id']
        intervention_pairs = ast.literal_eval(row['intervention_pairs'])
        
        for i in range(len(intervention_pairs)):
            speaker_text = intervention_pairs[i][0]
            intervention_text = intervention_pairs[i][1]
            
            new_row = {
                "session_id": session_id,
                "intervention_id": intervention_id,
                "speaker_text": speaker_text,
                "intervention_text": intervention_text
            }
            chunk_ints = pd.concat([chunk_ints, pd.DataFrame([new_row])], ignore_index=True)
            intervention_id += 1
    chunk_ints.to_csv(os.path.join(chunk_dir, f"interventions_chunk_{i}.csv"), index=False)

# join chunks, save intervention_df
intervention_df = pd.DataFrame(columns=["session_id", "intervention_id", "speaker_text", "intervention_text"])
for file_name in os.listdir(chunk_dir):
    file_path = os.path.join(chunk_dir, file_name)
    chunk = pd.read_csv(file_path)
    intervention_df = pd.concat([intervention_df, chunk], ignore_index=True)

intervention_df.to_csv(r"../outputs/interventions.csv", index=False)

# delete temp files
shutil.rmtree(chunk_dir)

# assign senator name

sen_urls = ['https://es.wikipedia.org/wiki/Categor%C3%ADa:Senadores_de_Colombia_2002-2006',
            'https://es.wikipedia.org/wiki/Categor%C3%ADa:Senadores_de_Colombia_2006-2010',
            'https://es.wikipedia.org/wiki/Categor%C3%ADa:Senadores_de_Colombia_2010-2014',
            'https://es.wikipedia.org/wiki/Categor%C3%ADa:Senadores_de_Colombia_2014-2018',
            'https://es.wikipedia.org/wiki/Categor%C3%ADa:Senadores_de_Colombia_2018-2022',
            'https://es.wikipedia.org/wiki/Categor%C3%ADa:Senadores_de_Colombia_2022-2026'
            ]

house_urls = ['https://es.wikipedia.org/wiki/Categor%C3%ADa:Representantes_de_la_C%C3%A1mara_de_Colombia_2002-2006',
              'https://es.wikipedia.org/wiki/Categor%C3%ADa:Representantes_de_la_C%C3%A1mara_de_Colombia_2006-2010',
              'https://es.wikipedia.org/wiki/Categor%C3%ADa:Representantes_de_la_C%C3%A1mara_de_Colombia_2010-2014',
              'https://es.wikipedia.org/wiki/Categor%C3%ADa:Representantes_de_la_C%C3%A1mara_de_Colombia_2014-2018',
              'https://es.wikipedia.org/wiki/Categor%C3%ADa:Representantes_de_la_C%C3%A1mara_de_Colombia_2018-2022',
              'https://es.wikipedia.org/wiki/Categor%C3%ADa:Representantes_de_la_C%C3%A1mara_de_Colombia_2022-2026']

missing_names = ['milene jarava díaz', 'germán rogelio rozo anís', 'jairo humberto cristo', 'etna tamara argote calderón', 'sor berenice bedoya pérez', 'norman david bañol álvarez',
                 'juan camilo londoño barrera', 'jorge alexander quevedo herrera', 'víctor manuel salcedo guerrero', 'nadya georgette blel scaff', 'yenica sugein acosta infante',
                 'maría eugenia lopera', 'jairo humberto cristo', 'germán josé gómez lópez', 'heráclito landínez suárez', 'ana carolina espitia jerez', 'guido echeverri piedrahíta',
                 'juan carlos vargas soler', 'leider alexandra vásquez ochoa', 'martha lisbeth alfonso jurado', 'julia miranda londoño', 'juan pablo celis vergel', 'jairo giovany cristancho tarache',
                 'andrea padilla villarraga', 'juan felipe corzo álvarez', 'héctor david chaparro chaparro', 'andrés eduardo forero molina', 'armando antonio zabaraín',
                 'betsy judith pérez arango', 'hugo alfonso archila suárez', 'juan sebastián gómez gonzález', 'edwin alberto valdés rodríguez', 'iván leonidas name vásquez',
                 'gabriel ernesto parrado durán', 'alfredo mondragón garzón', 'alfredo ape cuello baute', 'leyla marleny rincón trujillo', 'celis vergel juan pablo',
                 'jorge eliécer tamayo marulanda', 'josé octavio cardona león', 'julián peinado ramírez', 'josé eliécer salazar lópez', 'elizabeth jay-pang díaz',
                 'pedro josé suárez vacca', 'carlos alberto carreño marín', 'jairo humberto cristo correa', 'christian munir garcés aljure', 'luis miguel lópez aristizábal'
                 'duvalier sánchez arango', 'cristian danilo avendaño fino', 'juan daniel peñuela calvache', 'jorge eliécer tamayo marulanda', 'mary anne andrea perdomo gutiérrez:',
                 'astrid sánchez montes de oca', 'piedad correal rubiano', 'hugo danilo lozano pimiento', 'hernando gonzález', 'elkin rodolfo ospina', 'sandra yaneth jaimes cruz',
                 'gloria liliana rodríguez valencia', 'pedro hernando flórez porras', 'gustavo adolfo moreno hurtado', 'robert daza guevara', 'juana carolina londoño jaramillo',
                 'josé vicente carreño castro', 'hernando gonzález', 'josé vicente carreño castro', 'wadith alberto manzur imbett', 'wilmer yaír castellanos hernández',
                 'modesto enrique aguilera vides', 'julián david lópez tenorio', 'alexánder guarín silva', 'john jairo berrío lópez', 'fabián díaz plata', 'diela liliana benavides solarte',
                 'josé luis pinedo campo', 'nicolás albeiro echeverry alvarán', 'eduar alexis triana rincón', 'esteban quintero cardona', 'atilano alonso giraldo arboleda',
                 'diego javier osorio jiménez', 'álvaro henry monedero rivera', 'jénnifer kristín arias falla', 'héctor javier vergara sierra', 'mónica karina bocanegra pantoja',
                 'juana carolina londoño jaramillo', 'luis miguel lópez aristizábal', 'fernando david niño mendoza', 'edison vladimir olaya mancipe', 'william ferney aljure martínez',
                 'gerson lisímaco montaño arizala', 'cristóbal caicedo angulo', 'víctor manuel ortiz joya', 'edgar enrique palacio mizrahi', 'jonathan ferney pulido',
                 'dídier lobo chinchilla', 'sonia shirley bernal sánchez', 'gregorio eljach pacheco', 'john jairo bermúdez garcés', 'gabriel jaime vallejo', 'jhon arley murillo benítez',
                 'josé jaime uscátegui pastrana', 'josé daniel lópez jiménez', 'ángel maría gaitán pulido', 'abel david jaramillo largo', 'buenaventura león león', 'jorge dilson murcia olaya',
                 'dolcey óscar torres romero', 'lina maría garrido martín', 'haiver rincón gutiérrez', 'ronald valdés padilla', 'alexander tiles', 'leomaris evangelista',
                 'catalina del socorro pérez', 'beatriz lorena ríos cuéllar', 'jezmi lizeth barraza arraút', 'diego fernando caicedo navas', 'ingrid marlén sogamoso alfonso', 'karyme adrana cotes martínez',
                 'ermes evelio pete vivas', 'edward giovanny sarmiento hidalgo', 'hernando guida ponce', ' karen astrith manrique olarte', 'saray elena robayo bechara',
                 'john édgar pérez rojas', 'álvaro leonel rueda caballero', 'olga beatriz gonzales correa', 'eduard giovanny sarmiento hidalgo', 'diógenes quintero amaya', 'juan pablo salazar',
                 'ermes pete vivas', 'libardo cruz casado', 'yamil hernando arana padauí', 'erick adrián velasco burbano', 'james hermenegildo mosquera torres', 'jorge méndez hernández',
                 'gersel luis pérez altamiranda', 'gilma díaz arias', 'jorge rodrigo tovar vélez', 'carolina giraldo botero', 'neyla ruiz correa', 'john jairo hoyos garcía',
                 'élbert díaz lozano', 'mónica maría raigoza morales', 'milton hugo angulo viveros', 'henry fernando correal herrera', 'ángela patricia sánchez leal'
                 ]

sen_names = clean_list(get_names(sen_urls))
house_names = clean_list(get_names(house_urls))

names = list(set(sen_names + house_names + missing_names))

valid_int = []

for row in intervention_df.iterrows():
    if "honorable" in row[1]["speaker_text"] or "senador" in row[1]["speaker_text"] or "representante" in row[1]["speaker_text"]:
        valid_int.append(True)
    else:
        valid_int.append(False)

intervention_df['valid_int'] = valid_int

valid_ints = intervention_df[intervention_df['valid_int'] == True]

speaker = []
for row in valid_ints.iterrows():
    name = match_name(row[1]['speaker_text'], names)
    if name:
        speaker.append(name)
    else:
        speaker.append(False)

valid_ints['speaker'] = speaker

valid_ints.to_csv(r"../outputs/interventions_with_speakers.csv", index=False)