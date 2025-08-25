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
        print(f"❌ Error parsing {file_path}: {e}")
        return []

#  提取空间关系函数（保持不变）
def extract_spatial_relationships(text):
    prompt = f"""
You are an expert in spatial language and historical geography.

You are working with **historical writing from the 17th to 20th centuries** about the **Lake District**.

Your task is to extract only **spatial relationships** that are used to describe **scenic views, landscapes, or physical geography** of the Lake District.

A valid **spatial relationship** must:
- Involve two **physical entities** (such as natural or built features: lakes, mountains, valleys, roads, etc.)
- Include a clear **spatial relation** (e.g., "on", "in", "beside", "surrounded by", "runs through", "next to", "below", etc.)
- Depict how features are **positioned or arranged** in physical space (for example, how a road winds through a valley or a house sits on a hill)

  Examples of valid relationships:
    ("lake", "surrounded by", "mountains")
    ("village", "nestled in", "valley")
    ("path", "runs through", "forest")
    ("hill", "above", "the water")

  Do NOT include:
- Mentions of ownership or category (e.g. "lakes of the north", "part of the kingdom")
- Temporal or action-related phrases (e.g. "visited by", "routes by which travellers...")
- Non-spatial references (e.g. "named after", "known for", "written by")
- Abstract or metaphorical language not tied to physical geography

Focus only on physical arrangements that contribute to how the **Lake District’s scenery** is described.

Only output Python-style tuples in the form:

    ("Subject", "Spatial_Relation", "Object")

One per line. No other explanation or commentary.

Now extract the valid spatial relationships from this paragraph:

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
        print(f"❌ OpenAI API error: {e}")
        return ""


#  主逻辑：只处理一个文件
paragraphs = extract_paragraphs_from_xml(test_file)
results = []

for idx, paragraph in enumerate(paragraphs):
    extracted = extract_spatial_relationships(paragraph)

    if not extracted.strip():
        continue

    for line in extracted.strip().split("\n"):
    # 使用正则提取括号内的部分
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
            print(f"⚠️ 解析失败：{line} → {e}")

#  保存为 CSV（只处理一个文件）
df = pd.DataFrame(results)
df.to_csv("test3_singlefile.csv", index=False)
print(" Done! Results saved to test3_singlefile.csv")