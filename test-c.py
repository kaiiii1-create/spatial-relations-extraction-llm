import os
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
import xml.etree.ElementTree as ET
import pandas as pd
import re
client = OpenAI()


test_file = "LakeDistrictCorpus/LD80_transcribed/Anon1857_b.xml"


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
        print(f" Error parsing {file_path}: {e}")
        return []


def extract_spatial_relationships(text):
    prompt = f"""
Your task is to extract only explicit spatial relationships from the given paragraph.

Output format:

   Output Python tuples only in the form:("Subject","Relation","Object")
   One tuple per line, in reading order.
   Use the exact surface strings from the text.
   If no valid relation is found, output nothing.

Scope of entities:
   Subject and Object must both be physical geographic entities such as:
       places, settlements, buildings, ruins, rivers, lakes, islands, hills, mountains, valleys, shores, promontories, roads, bridges.
    Exclude abstract concepts, temporal expressions, directions, measurements, or descriptive phrases that are not physical entities.

Excluded categories (to be filtered）:

   These expressions must never be treated as spatial relations:
        Naming/Identity: any form of “to be called/named/known as” (e.g. called, is called, are called, was called, named, being known as)
        Perception/Viewpoint: any form of “see/view/prospect/look over” (e.g. is seen, are seen, view of, views of, prospect, prospects, looks over)
        Comparison/Degree: any comparative or superlative expression (e.g. as … as, larger than, smaller than, more … than, less … than, highest, lowest)
        Attribute/Measurement: any expression describing length, height, area, population, distance, size, number
        Judgement/Equivalence: is, are (when used as “X is Y” type of statements, e.g., naming, definition, classification)

Procedure:
   1.Identify candidate geographic entities in the paragraph (places, settlements, buildings, rivers, lakes, hills, valleys, shores, roads).
   2.Ensure both Subject and Object are physical entities (discard if either is abstract).
   3.Exclude any candidates matching the forbidden categories above.
   4.Output tuples exactly as specified; avoid duplicates within the paragraph.

Paragraph:

\"\"\"{text}\"\"\"
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini", 
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f" OpenAI API error: {e}")
        return ""



paragraphs = extract_paragraphs_from_xml(test_file)
results = []

for idx, paragraph in enumerate(paragraphs):
    extracted = extract_spatial_relationships(paragraph)
    print(f"\n--- GPT Output for Paragraph {idx + 1} ---\n{extracted}\n")

    if not extracted.strip():
        continue

    for line in extracted.strip().split("\n"):
    
        match = re.search(r'\(\s*[\'"]?(.*?)[\'"]?\s*,\s*[\'"]?(.*?)[\'"]?\s*,\s*[\'"]?(.*?)[\'"]?\s*\)', line)
        if match:
           try:
            subject, relation, obj = match.groups()
            results.append({
                "File": os.path.basename(test_file),
                "ParagraphID": idx + 1,
                "ParagraphText": paragraph,
                "Subject": subject.strip(),
                "Relation": relation.strip(),
                "Object": obj.strip()
            })
           except Exception as e:
            print(f" failed：{line} → {e}")


df = pd.DataFrame(results)
df.to_csv("test-c_singlefile.csv", index=False)
print(" Done! Results saved to test-c_singlefile.csv")