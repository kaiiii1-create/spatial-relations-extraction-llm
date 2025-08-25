import os
import pandas as pd
import re
import time
from openai import OpenAI
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()
client = OpenAI()

df = pd.read_csv("paragraphs_all.csv")
results = []

def extract_spatial_relationships(text):
    prompt = f"""
Extract only explicit spatial relationships from the paragraph.

Rules:
- Output Python tuples only: ("Subject","Relation","Object"), one per line.
- Use exact surface strings.
- Relation must be an explicit spatial cue (e.g., in, on, at, near, between, along, across, opposite,
  north of/south of/east of/west of, upstream, downstream, ends at, flows into, runs along, spans, stretches into).
- Subject and Object must both be physical geographic entities (settlements, buildings, ruins, rivers, lakes,
  islands, hills, mountains, valleys, shores, promontories, roads, bridges).
- Do NOT output naming/identity, perception/viewpoint, comparison/degree, measurement, or ornamental/covering expressions.
- If none, or if uncertain, output nothing.

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

Paragraph: The bridge spans the river.
Output:
("The bridge","spans","the river")

Paragraph: The road ends at Seathwaite.
Output:
("The road","ends at","Seathwaite")

Invalid:
Paragraph: The group of curious stones is called Long Meg and her Daughters.
Output:
# nothing

Paragraph: Hen Holm is in view of Windermere.
Output:
# nothing

Paragraph: Ulleswater is nine miles in length.
Output:
# nothing

Paragraph: The town is beautiful and adorned with gardens.
Output:
# nothing

Paragraph: The castle is larger than the church.
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
        return f"__ERROR__: {e}"

def process_row(row):
    file_name = row["File"]
    para_id = row["ParagraphID"]
    paragraph = row["ParagraphText"]

    extracted = extract_spatial_relationships(paragraph)

    output = []
    if "__ERROR__" in extracted or not extracted.strip():
        return output

    for line in extracted.strip().split("\n"):
        match = re.search(r"\(([^)]+)\)", line)
        if match:
            try:
                subject, relation, obj = eval(f"({match.group(1)})")
                output.append({
                    "File": file_name,
                    "ParagraphID": para_id,
                    "ParagraphText": paragraph,
                    "Subject": subject.strip(),
                    "Relation": relation.strip(),
                    "Object": obj.strip()
                })
            except:
                continue
    return output

MAX_WORKERS =9 
start = time.time()

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = [executor.submit(process_row, row) for _, row in df.iterrows()]
    for i, future in enumerate(as_completed(futures)):
        try:
            results.extend(future.result())
            print(f" Finished {i+1}/{len(df)}")
        except Exception as e:
            print(f" Failed a task: {e}")

#  save
df_out = pd.DataFrame(results)
df_out.to_csv("extractall_t1.csv", index=False)
print(f" All done in {round(time.time() - start, 2)} seconds. Saved to extractall_t1.csv.")
