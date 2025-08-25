import os, requests
from lxml import etree
from tqdm import tqdm

# 中文注释：配置 Ollama 接口和模型名称
# English: Configure Ollama API and model name
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma:2b-instruct"

# 中文注释：XML 文件夹路径
# English: Path to the XML document folder
XML_DIR = "LakeDistrictCorpus/LD80_transcribed"

# 中文注释：用于存储结果的CSV文件路径
# English: Output CSV file path
OUT_FILE = "spatial_relationships_output.csv"

# 中文注释：用于提取文本内容的函数
# English: Function to extract all text from XML
def extract_text_from_xml(xml_path):
    try:
        tree = etree.parse(xml_path)
        text = " ".join(tree.xpath('//text()'))
        return text.strip()
    except Exception as e:
        print(f"Error parsing {xml_path}: {e}")
        return ""

# 中文注释：使用 Ollama 调用 gemma 模型提取空间关系
# English: Use Ollama with gemma model to extract spatial relationships
def get_spatial_relationships(text):
    prompt = (
    "From the following text, extract all spatial relationships. "
    "For each spatial relationship, identify:\n"
    "- the full sentence where it occurs,\n"
    "- the first object,\n"
    "- the spatial relationship phrase,\n"
    "- the second object.\n"
    "Output the results as a CSV-style table with the following columns: Sentence, Object1, Relationship, Object2.\n\n"
    f"Text:\n{text}\n\n"
    "Output format:\nSentence,Object1,Relationship,Object2"
)

    
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }
    
    try:
        response = requests.post(OLLAMA_URL, json=payload)
        if response.status_code == 200:
            return response.json().get("response", "").strip()
        else:
            return f"Error {response.status_code}"
    except Exception as e:
        return f"Request failed: {e}"

# 中文注释：主程序，遍历所有 XML 文件提取空间关系
# English: Main script to iterate through XML files and extract spatial relationships
results = []

print(" Extracting spatial relationships from XML files...")

for filename in tqdm(os.listdir(XML_DIR)):
    if filename.endswith(".xml"):
        xml_path = os.path.join(XML_DIR, filename)
        text = extract_text_from_xml(xml_path)
        if text:
            response = get_spatial_relationships(text[:1000])  # 可根据需要调整长度
            results.append((filename, response))

# 中文注释：保存结果到CSV文件
# English: Save results to a CSV file
import csv
with open(OUT_FILE, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Filename", "Extracted_Spatial_Relationships"])
    writer.writerows(results)

print(f" Finished! Results saved to {OUT_FILE}")
