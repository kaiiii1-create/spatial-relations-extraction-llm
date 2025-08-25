import os
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
import xml.etree.ElementTree as ET
import pandas as pd
import re
client = OpenAI()

# ✅ 你想测试的 XML 文件路径
test_file = "LakeDistrictCorpus/LD80_transcribed/Anon1857_b.xml"

# ✅ 提取段落函数（保持不变）
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

# ✅ 提取空间关系函数（保持不变）
def extract_spatial_relationships(text):
    prompt = f"""
You are an expert in spatial language understanding.

Your task is to extract only spatial relationships from the paragraph below.

A spatial relationship should describe how two physical things (such as hills, lakes, trees, buildings) are arranged in space.

Focus especially on spatial relationships that are used to describe scenery and views, as perceived or described by the author.

Only include meaningful spatial arrangements — not actions, events, names, or temporal references.

Output a list of Python-style (Subject, Spatial_Relation, Object) tuples, one per line. No explanations.

Paragraph:

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
        print(f"❌ OpenAI API error: {e}")
        return ""


# ✅ 主逻辑：只处理一个文件
paragraphs = extract_paragraphs_from_xml(test_file)
results = []

for idx, paragraph in enumerate(paragraphs):
    print(f"🧠 Extracting from paragraph {idx + 1}/{len(paragraphs)}")
    extracted = extract_spatial_relationships(paragraph)
    print(f"🧾 GPT Response for Paragraph {idx + 1}:\n{extracted}\n{'-'*50}")

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

# ✅ 保存为 CSV（只处理一个文件）
df = pd.DataFrame(results)
df.to_csv("test6_singlefile.csv", index=False)
print("✅ Done! Results saved to test6_singlefile.csv")

