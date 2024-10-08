import PyPDF2
import os
import fitz
import pytesseract
from PIL import Image
# api_key = [輸入你的金鑰]
# prompt = [你想要給的文字]

# response = requests.post(
#     'https://api.openai.com/v1/completions',
#     headers = {
#         'Content-Type': 'application/json',
#         'Authorization': f'Bearer {api_key}'
#     },
#     json = {
#         'model': 'text-davinci-003',
#         'prompt': prompt,
#         'temperature': 0.4,
#         'max_tokens': 300
#     })

# #使用json解析
#json = response.json()
#print(json)
finance_path="reference/finance"
def pdf_to_text(file_path):
    # Open the PDF file
    doc = fitz.open(file_path)

    extracted_text = ''

    # Loop through each page in the PDF
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        pix = page.get_pixmap()  # Convert page to an image

        # Convert pixmap to PIL Image for OCR
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        # Extract text using OCR
        extracted_text += pytesseract.image_to_string(img,lang='chi_tra+eng')

    # Print or save the extracted text
    print(extracted_text)
    return extracted_text
# Example usage
file_path = os.path.join(finance_path,"1.pdf")
pdf_text = pdf_to_text(file_path)
print(len(pdf_text))
