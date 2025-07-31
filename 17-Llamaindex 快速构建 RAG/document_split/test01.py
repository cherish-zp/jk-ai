from llama_index.core import SimpleDirectoryReader

reader = SimpleDirectoryReader(
    input_files=["/home/cw/projects/demo_17/data/README_zh-CN.md"]
)

# reader = SimpleDirectoryReader(
#     "/home/cw/projects/demo_20/data"
# )
docs = reader.load_data()
print(f"Loaded {len(docs)} docs")
print(docs)

# # 案例2：高级解析
# import pdfplumber

# with pdfplumber.open("/home/cw/projects/demo_20/data/report_with_table.pdf") as pdf:
#     # 提取所有文本
#     text = ""
#     for page in pdf.pages:
#         text += page.extract_text()
#     print(text[:200])  # 打印前200字符

#     # 提取表格（自动检测）
#     for page in pdf.pages:
#         tables = page.extract_tables()
#         for table in tables:
#             print("\n表格内容：")
#             for row in table:
#                 print(row)