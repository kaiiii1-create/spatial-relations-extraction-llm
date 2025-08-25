import os
import xml.etree.ElementTree as ET
import pandas as pd

xml_folder = "LakeDistrictCorpus/LD80_transcribed"

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

results = []

for filename in os.listdir(xml_folder):
    if filename.endswith(".xml"):
        file_path = os.path.join(xml_folder, filename)
        print(f"Processing: {filename}")
        paragraphs = extract_paragraphs_from_xml(file_path)

        for idx, paragraph in enumerate(paragraphs):
            results.append({
                "File": filename,
                "ParagraphID": idx + 1,
                "ParagraphText": paragraph
            })

#  save to CSV
df = pd.DataFrame(results)
df.to_csv("paragraphs_all.csv", index=False)
print("Done! All paragraphs saved to paragraphs_all.csv")
