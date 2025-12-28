# 学习 markitdown: 
# png => md (OCR 不工作)
# pdf => md (列表、表格等格式丢失，内容可以)

# import easyocr
from markitdown import MarkItDown

# png_path = "../temp/t1.png"
# reader = easyocr.Reader(['ch_sim', 'en'], gpu=False)
# results = reader.readtext(png_path, detail=0, paragraph=True)
# text = "\n".join(results)
# print(text)

md = MarkItDown(enable_plugins=True, lign_tables=True, preserve_newlines=True)
# 支持 .docx, .xlsx, .pptx, .html, .zip, .png/.jpg, .mp3 等
result = md.convert("../temp/t1.pdf")
# text = getattr(result, "text_content", None)
# if isinstance(text, str) and text.strip():
#     print(text)
with open("output.md", "w", encoding="utf-8") as f:
    f.write(result.text_content)
# print(result.text_content)
