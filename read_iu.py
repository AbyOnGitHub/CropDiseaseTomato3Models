import os

sp = r"C:\Users\abbsw\AppData\Roaming\Python\Python312\site-packages\transformers\utils\import_utils.py"
with open(sp, "r", encoding="utf-8") as f:
    lines = [f.readline() for _ in range(70)]
for idx, l in enumerate(lines):
    print(f"{idx+1}: {l.rstrip()}")
