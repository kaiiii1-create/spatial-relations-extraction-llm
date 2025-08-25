import os
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
import xml.etree.ElementTree as ET
import pandas as pd
import re
client = OpenAI()

#  你想测试的 XML 文件路径
test_file = "LakeDistrictCorpus/LD80_transcribed/Anon1857_b.xml"

#  提取段落函数（保持不变）
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

#  提取空间关系函数（保持不变）
def extract_spatial_relationships(text):
    prompt = f"""
You are an expert in spatial language understanding, analyzing Lake District writings from the 17th to 20th centuries.

Your task is to extract only spatial relationships — expressions that describe how physical features or locations are arranged in space.

Each spatial relationship must be written as a (Subject, Spatial_Relation, Object) tuple.

Strict constraints:

1. Only include relationships that are explicitly stated in the paragraph. Do not infer, assume, or imagine any spatial relations that are not directly written.
2. Each subject and object must refer to **concrete physical entities**, such as rivers, lakes, hills, valleys, fields, forests, buildings, bridges, roads, mountains, etc.
3. The Spatial_Relation must be a **core spatial preposition or phrase**, such as: in, on, over, beside, near, under, along, through, across, around, between, downstream, upstream, at the foot of, located in, surrounded by, etc.
   - Do not include auxiliary or linking verbs like “is”, “was”, “are”, “were”. For example:
     - Input: "Belle Isle is near Bowness."
     - Output: ("Belle Isle", "near", "Bowness")
4. Do not include any of the following:
   - Classification or membership statements:
     - e.g. "X is one of the mountains that...", "X is part of...", "X belongs to..."
   - Motion or action verbs:
     - e.g. went, passed, built, traveled, joins, diverges, leads to, etc.
   - Identity or naming relations:
     - e.g. is, are, called, named, known as, described as
     - Incorrect example:
       - Input: "A lake called Buttermere lies in the valley."
       - Do not output: ("lake", "called", "Buttermere")
       - Because "called" is a naming verb, not a spatial relation.
   - Viewpoint or perceptual language:
     - e.g. "from which", "offers a view of", "seen from"
   - Temporal or event-based expressions
   - Abstract or emotional descriptions (e.g. beauty, silence, fear)
5. If a phrase is ambiguous or vague and does not clearly express a spatial relationship, skip it.
6. If either the subject or object is missing, unclear, or metaphorical, do not generate a tuple.
7. If no valid spatial relationships are found, return an empty list: []

Your output must be a clean list of tuples only. Do not include any commentary, explanation, or headings.

Now extract the valid spatial relationships from this paragraph:

\"\"\"{text}\"\"\"
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # or gpt-4o if you have access
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f" OpenAI API error: {e}")
        return ""


#  主逻辑：只处理一个文件
paragraphs = extract_paragraphs_from_xml(test_file)
results = []

for idx, paragraph in enumerate(paragraphs):
    extracted = extract_spatial_relationships(paragraph)
    print(f"\n--- GPT Output for Paragraph {idx + 1} ---\n{extracted}\n")

    if not extracted.strip():
        continue

    for line in extracted.strip().split("\n"):
    # 使用正则提取括号内的部分
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
            print(f" 解析失败：{line} → {e}")

#  保存为 CSV（只处理一个文件）
df = pd.DataFrame(results)
df.to_csv("test11_singlefile.csv", index=False)
print(" Done! Results saved to test11_singlefile.csv")

