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
Extract only explicit spatial relationships from the paragraph.

Rules:
- Output Python tuples only: ("Subject","Relation","Object"), one per line.
- Use exact surface strings.
- Subject/Object must be physical geographic entities (places, settlements, buildings, ruins, rivers, lakes, islands, hills, mountains, valleys, shores, promontories, roads, bridges).
- Do not output naming, identity, perception, comparison, measurement, or ornamental/covering expressions.
- If none, output nothing.

Examples

 Valid:
Paragraph: The valley lies between two mountains.
Output:
("The valley","between","two mountains")

Paragraph: Friars Crag stretches out into the lake.
Output:
("Friars Crag","stretches out into","the lake")

Paragraph: The stream flows into the river.
Output:
("The stream","flows into","the river")

 Invalid:
Paragraph: The group of curious stones is called Long Meg and her Daughters.
Output:
# nothing

Paragraph: Hen Holm is in view of Windermere.
Output:
# nothing

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
df.to_csv("test-d_singlefile.csv", index=False)
print(" Done! Results saved to test-d_singlefile.csv")