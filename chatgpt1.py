import os
from dotenv import load_dotenv
load_dotenv()

import openai
from openai import OpenAI
import pandas as pd
import xml.etree.ElementTree as ET

##openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()
xml_folder = "LakeDistrictCorpus/LD80_transcribed/Anon1857_b.xml"

def extract_paragraphs_from_xml(file_path):
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        paragraphs = []
        for p in root.iter("p"):
            full_text = ''.join(p.itertext()).strip()
            if len(full_text) > 10:
                paragraphs.append(full_text)
        return paragraphs
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return []


def extract_spatial_relationships(text):
    prompt = f"""
You are an expert in spatial language understanding.
Extract all spatial relationships from the following English paragraph.
Return a list of (Subject, Spatial_Relation, Object) triples.

Text:
\"\"\"{text}\"\"\"

Only return a list of such triples.
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"OpenAI API error: {e}")
        return ""


results = []


for filename in os.listdir(xml_folder):
    if not filename.endswith(".xml"):
        continue

    file_path = os.path.join(xml_folder, filename)
    print(f"Processing: {filename}")
    paragraphs = extract_paragraphs_from_xml(file_path)

    for idx, paragraph in enumerate(paragraphs):
        print(f"Paragraph {idx + 1}/{len(paragraphs)}")

        extracted = extract_spatial_relationships(paragraph)

        for line in extracted.strip().split("\n"):
            if line.startswith("(") and line.endswith(")"):
                try:
                    subject, relation, obj = eval(line)
                    results.append({
                        "File": filename,
                        "ParagraphID": idx + 1,
                        "ParagraphText": paragraph,
                        "Subject": subject,
                        "Relation": relation,
                        "Object": obj
                    })
                except:
                    continue


df = pd.DataFrame(results)
df.to_csv("chatgpt1_test.csv", index=False)
print("successful! chatgpt1_test.csv")
