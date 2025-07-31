from llama_index.readers.file import HTMLTagReader

reader = HTMLTagReader(tag="section", ignore_no_id=True)
docs = reader.load_data(
    "/home/cw/projects/demo_20/V8W9yJZ/index.html"
)

for doc in docs:
    print(doc.metadata)

print(docs)