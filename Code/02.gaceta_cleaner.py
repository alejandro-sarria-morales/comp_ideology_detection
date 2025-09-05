import pandas as pd
import os
import re
import datetime
import pymupdf
import unicodedata
import nltk
from nltk.corpus import stopwords
import spacy
from tqdm import tqdm
import ftfy

nlp = spacy.load("es_core_news_sm")
nltk.download('stopwords')
span_stopwords = stopwords.words('spanish')

def remove_noise_chars(text):
    text = re.sub(r"(\\x[0-9a-fA-F]{2}|/x[0-9a-fA-F]{2})", "", text)
    return re.sub(r"[^a-zA-Z0-9áéíóúüñÁÉÍÓÚÜÑ.,:;()\"'¿?¡!\-* \n]", "", text)

def remove_invisible_chars(text):
    invisible_chars = [
        '\u00ad', '\u200b', '\u200c', '\u200d',
        '\u200e', '\u200f', '\ufeff', '\u2028', '\u2029'
    ]
    pattern = f"[{''.join(invisible_chars)}]"
    return re.sub(pattern, '', text)

def extract_text_with_bold(filename):
    doc = pymupdf.open(filename)
    extracted_text = ""

    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"]
                        text = unicodedata.normalize("NFKC", text)
                        text = ftfy.fix_text(text)
                        text = remove_invisible_chars(text)
                        text = remove_noise_chars(text)

                        if span["font"].lower().find("bold") >= 0:
                            extracted_text += f"*b-*{text}*-b* "
                        else:
                            extracted_text += text + " "
                    extracted_text += "\n"
            extracted_text += "\n"

    doc.close()
    return extracted_text

def clean_raw_text(text):
    clean_text = re.sub(r"IMPRENTA\s*NACIONAL\s*DE\s*COLOMBIA.*|www\.\w+\.gov\.co", "", text, flags=re.IGNORECASE)
    clean_text = re.sub(r"Página\s*\d+|Edición\s*de.*páginas", "", clean_text, flags=re.IGNORECASE)
    clean_text = re.sub(r"Año.*Nº.*", "", clean_text, flags=re.IGNORECASE)
    clean_text = "\n".join([line for line in clean_text.splitlines() if re.search(r'[a-z]', line)])
    clean_text = re.sub(r"\n\s*\n", "\n", clean_text).strip()
    clean_text = re.sub(r"- \n", "", clean_text)
    clean_text = re.sub(r"\*-b\*\s*\n\s*\*b-\*", " ", clean_text)
    clean_text = re.sub(r"\*-b\*", r"\n*-b*\n", clean_text)
    clean_text = re.sub(r"\*b-\*", r"\n*b-*\n", clean_text)
    clean_text = re.sub(r"\n\s*\n", "\n", clean_text).strip()
    clean_text = re.sub(r"[ ]{2,}", " ", clean_text).strip()
    clean_text = re.sub(r"\n", " ", clean_text).strip()
    clean_text = re.sub(r"^.*?ACTA NÚMERO \d+ DE \d+", r"", clean_text, count=1, flags=re.DOTALL)[5:]
    clean_text = re.sub(r"\*b-.{0,3}-b\*", "", clean_text)
    return clean_text.strip().lower()

def remove_header(text: str) -> str:
    pattern = r'''(?i)(?:\b(?:lunes|martes|miércoles|miercoles|jueves|viernes|sábado|sabado|domingo),\s*\d{1,2}\s+de\s+[a-záéíóú]+\s+de\s+\d{4}\s+gaceta del congreso\s+\d+|
                  gaceta del congreso\s+\d+\s+\b(?:lunes|martes|miércoles|miercoles|jueves|viernes|sábado|sabado|domingo),\s*\d{1,2}\s+de\s+[a-záéíóú]+\s+de\s+\d{4})'''
    return re.sub(pattern, "", text, flags=re.VERBOSE)

def extract_session_info(raw_text):
    months_spanish = ["enero", "febrero", "marzo", "abril", "mayo", "junio", 
                      "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]
    date_pattern = r"(\d{1,2})\s+de\s+(" + "|".join(months_spanish) + r")\s+de\s+(\d{4})"
    match = re.search(date_pattern, raw_text, re.IGNORECASE)
    if match:
        date = datetime.date(int(match.group(3)), months_spanish.index(match.group(2)) + 1, int(match.group(1)))
    else:
        date = None

    header = re.sub(r"\s", "", raw_text[:1000]).upper()
    chamber = "house" if "CÁMARADEREPRESENTANTES" in header else "senate"
    instance = "commitee" if "COMISIÓN" in header or "COMISION" in header else "plenary"

    return date, chamber, instance 

def extract_headline_intervention_pairs(text):
    pairs = []
    headline = None
    segments = re.split(r'(\*b-\*.*?\*-b\*)', text, flags=re.DOTALL)

    for segment in segments:
        if segment.startswith('*b-*') and segment.endswith('*-b*'):
            current_headline = segment.replace('*b-*', '').replace('*-b*', '').strip().lower()
            if 'proyecto' not in current_headline and len(current_headline) > 5:
                if headline is not None:
                    if len(intervention.strip()) < 5:
                        headline += ' ' + current_headline
                    else:
                        pairs.append((headline, intervention.strip()))
                        headline = current_headline
                else:
                    headline = current_headline
                intervention = ""
        else:
            if headline is not None:
                if len(segment.strip()) < 5:
                    headline += ' ' + segment.strip()
                else:
                    intervention += segment.strip() + " "
    
    if headline is not None and intervention:
        pairs.append((headline, intervention.strip()))

    return pairs

def tokenize_intervention(int_pairs):
    tokenized_interventions = []
    for _, intervention in int_pairs:
        doc = nlp(intervention)
        tokens = [
            token.lemma_.lower()
            for token in doc
            if not token.is_stop
            and not token.is_punct
            and not token.like_num
            and token.is_alpha
        ]
        tokenized_interventions.append(tokens)
    return tokenized_interventions

def process_pdf(file_path): 
    raw_text = extract_text_with_bold(file_path)
    clean_text = clean_raw_text(raw_text)
    clean_text = remove_header(clean_text)
    info = extract_session_info(clean_text) 
    gaceta_numb = os.path.basename(file_path)[:-4]
    intervention_pairs = extract_headline_intervention_pairs(clean_text)

    return pd.DataFrame({
        "gaceta_numb": gaceta_numb, 
        "date": info[0], 
        "chamber": info[1], 
        "type": info[2], 
        "raw_text": raw_text,
        "clean_text": clean_text,
        "intervention_pairs": [intervention_pairs]},
        index=[0])

if __name__ == "__main__":
    folder = r"D:\Thesis\sessions\raw_files"
    destination = r"C:\Users\asarr\Documents\Projects\comp_ideology_detection\outputs\sessions.csv"

    df = pd.DataFrame(columns=["id", "date", "chamber", "type", "raw_text", "clean_text", "intervention_pairs"])

    for file_name in tqdm(os.listdir(folder)):
        file_path = os.path.join(folder, file_name)
        try:
            new_row = process_pdf(file_path)
        except Exception as e:
            print(f"error in file {file_name}")
            print(e)
            continue
        df = pd.concat([df, new_row], ignore_index=True)

    df["id"] = df.index + 1

    df.to_csv(destination)
