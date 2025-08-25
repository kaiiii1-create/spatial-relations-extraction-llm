import os
import json
import xml.etree.ElementTree as ET
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

# ---------- 1. 读取段落 ----------
def extract_paragraphs_from_xml(file_path):
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        paragraphs = []
        for p in root.iter("p"):
            text = ''.join(p.itertext()).strip()
            if len(text) > 10:
                paragraphs.append(text)
        return paragraphs
    except Exception as e:
        print(f" Error reading XML: {e}")
        return []

# ---------- 2. 判断关系是否合法 ----------
def is_valid_spatial_relation(relation):
    relation = relation.lower().strip()
    invalid_keywords = [
        "called", "named", "known as", "described as",
        "is", "are", "was", "were", "belongs to", "one of"
    ]
    return not any(bad in relation for bad in invalid_keywords)

# ---------- 3. Function schema ----------
function_schema = {
    "name": "extract_spatial_relationship",
    "description": "Extract one spatial relationship from the text, in (subject, spatial_relation, object) format.",
    "parameters": {
        "type": "object",
        "properties": {
            "subject": {"type": "string"},
            "spatial_relation": {
                "type": "string",
                "description": "Spatial preposition or phrase. Avoid verbs like called, named, is, are, etc."
            },
            "object": {"type": "string"}
        },
        "required": ["subject", "spatial_relation", "object"]
    }
}

# ---------- 4. 提取关系 ----------
def extract_relation_from_text(text):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # 支持 function calling 的模型
            messages=[
                {"role": "system", "content": "You are an expert in extracting spatial relationships between physical places."},
                {"role": "user", "content": f"Extract one spatial relationship from this paragraph:\n\n{text}"}
            ],
            functions=[function_schema],
            function_call={"name": "extract_spatial_relationship"},
            temperature=0
        )
        args_str = response.choices[0].message.function_call.arguments
        result = json.loads(args_str)

        if is_valid_spatial_relation(result["spatial_relation"]):
            return result
        else:
            return None
    except Exception as e:
        print(f" OpenAI API error: {e}")
        return None

# ---------- 5. 主程序 ----------
def run_spatial_extraction(file_path, output_csv):
    paragraphs = extract_paragraphs_from_xml(file_path)
    filename = os.path.basename(file_path)
    extracted = []

    for idx, para in enumerate(paragraphs):
        result = extract_relation_from_text(para)
        if result:
            extracted.append({
                "File": filename,
                "ParagraphID": idx + 1,
                "ParagraphText": para,
                "Subject": result["subject"].strip(),
                "Relation": result["spatial_relation"].strip(),
                "Object": result["object"].strip()
            })

    df = pd.DataFrame(extracted)
    df.to_csv(output_csv, index=False)
    print(f"\n 提取完成，共 {len(extracted)} 条关系，已保存到：{output_csv}")

# ---------- 执行 ----------
if __name__ == "__main__":
    input_xml = "LakeDistrictCorpus/LD80_transcribed/Anon1857_b.xml"  
    output_csv = "test12_singlefile.csv"
    run_spatial_extraction(input_xml, output_csv)
