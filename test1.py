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
You are an expert in spatial language understanding.

Your task is to extract all spatial relationships from the paragraph below.

A spatial relationship should be represented as a (Subject, Spatial_Relation, Object) tuple.

Return only a list of such Python-style tuples, one per line. No commentary.

For example, from this sentence:

    "A lake surrounded by trees, a hill next to a forest, and a lake dotted with islands."

You should output:

    ("lake", "surrounded by", "trees")
    ("hill", "next to", "forest")
    ("lake", "dotted with", "islands")

Now extract spatial relationships from this paragraph:

\"\"\"{text}\"\"\"
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # or gpt-4o if you have access
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
    print(f" Extracting from paragraph {idx + 1}/{len(paragraphs)}")
    extracted = extract_spatial_relationships(paragraph)
    print(f" GPT Response for Paragraph {idx + 1}:\n{extracted}\n{'-'*50}")

    if not extracted.strip():
        continue

    for line in extracted.strip().split("\n"):
        match = re.search(r"\(([^)]+)\)", line)
        if match:
           try:
            subject, relation, obj = eval(f"({match.group(1)})")
            results.append({
                "File": os.path.basename(test_file),
                "ParagraphID": idx + 1,
                "ParagraphText": paragraph,
                "Subject": subject.strip(),
                "Relation": relation.strip(),
                "Object": obj.strip()
            })
           except Exception as e:
            print(f" failure：{line} → {e}")

df = pd.DataFrame(results)
df.to_csv("test1_singlefile.csv", index=False)
print(" Done! Results saved to test1_singlefile.csv")

