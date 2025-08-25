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

# ✅ 高置信度空间关键词，用于过滤判断
SPATIAL_KEYWORDS = {
    "in", "on", "at", "over", "under", "above", "below", "near", "next to", "beside", "between",
    "inside", "outside", "around", "through", "across", "along", "to", "from", "toward", "away from",
    "into", "onto", "out of", "off", "up", "down", "opposite", "across from", "surrounded by",
    "enclosed by", "covered with", "filled with", "lined with", "dotted with", "embedded in",
    "adjacent to", "bordering", "flanked by", "nestled in", "located in", "situated on", "perched on",
    "built into", "rests on", "occupies", "at the foot of", "at the top of", "at the base of",
    "downstream from", "upstream of", "on the slope of", "along the ridge", "across the valley",
    "facing", "behind", "in front of", "underneath", "beyond", "alongside", "at the corner of"
}

def is_spatial_relation(relation):
    """🧹 判断是否为可信空间关系"""
    return relation.lower().strip() in SPATIAL_KEYWORDS

# ✅ 从 XML 中提取段落
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

# ✅ 调用 GPT 提取空间关系
def extract_spatial_relationships(text):
    prompt = f"""
You are an expert in spatial language understanding.

The following paragraph is a piece of Lake District writing from the 17th to the 20th century.

Your task is to extract only spatial relationships that describe how physical locations are situated relative to one another in space — especially how spatial terms are used to depict **scenic views** of the Lake District.

A **spatial relationship** must be a (Subject, Spatial_Relation, Object) tuple, where:
- Subject and Object are **real-world physical entities**
- Spatial_Relation clearly describes the physical positioning of Subject relative to Object

✅ Valid examples:
    ("lake", "surrounded by", "trees")
    ("bridge", "over", "river")
    ("village", "in", "valley")
    ("road", "runs along", "shore")

❌ Do NOT include:
- Temporal or event-based descriptions (e.g., "visited by", "route from", "known as")
- Abstract or metaphorical relations
- Mere naming relationships
- Anything without a clear spatial keyword

Return only valid Python-style tuples, one per line. No explanation.

\"\"\"{text}\"\"\"
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # or gpt-4o-mini
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"❌ OpenAI API error: {e}")
        return ""

# ✅ 主程序（处理单个文件）
paragraphs = extract_paragraphs_from_xml(test_file)
results = []

for idx, paragraph in enumerate(paragraphs):
    print(f"⏳ Extracting from paragraph {idx+1}/{len(paragraphs)}...")
    extracted = extract_spatial_relationships(paragraph)
    print(f"GPT result:\n{extracted}\n{'-'*50}")

    if not extracted.strip():
        continue

    for line in extracted.strip().split("\n"):
        match = re.search(r"\(([^)]+)\)", line)
        if match:
            try:
                subject, relation, obj = eval(f"({match.group(1)})")

                # ✅ 过滤不在关键词列表中的 Relation
                if is_spatial_relation(relation):
                    results.append({
                        "File": os.path.basename(test_file),
                        "ParagraphID": idx + 1,
                        "ParagraphText": paragraph,
                        "Subject": subject.strip(),
                        "Relation": relation.strip(),
                        "Object": obj.strip()
                    })
                else:
                    print(f"🧹 Skipped non-spatial: ({subject}, {relation}, {obj})")

            except Exception as e:
                print(f"⚠️ Parse failed: {line} → {e}")

# ✅ 保存结果为 CSV
df = pd.DataFrame(results)
df.to_csv("test4_singlefile.csv", index=False)
print("✅ Done! Results saved to test4_singlefile.csv")
