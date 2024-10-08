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
pdf_path=["reference/finance","reference/insurance"]
def pdf_to_text(file_path):
    # Open the PDF file
    doc = fitz.open(file_path)

    extracted_text = ''
    zoom_x = 2.0  
    zoom_y = 2.0  
    mat = fitz.Matrix(zoom_x, zoom_y)  
    # Loop through each page in the PDF
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(matrix=mat)  # Convert page to an image

        # Convert pixmap to PIL Image for OCR
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        # Extract text using OCR
        extracted_text += pytesseract.image_to_string(img,lang='chi_tra+eng')

    # Print or save the extracted text
    
    return extracted_text
# Example usage
for path in pdf_path:
    file_list=os.listdir(path)
    file_list=sorted(file_list)
    for file in file_list:
        id,ext= os.path.splitext(file)
        file_path = os.path.join(path,file)
        pdf_text = pdf_to_text(file_path)
        text_path="reference/finance_text"
        if(not os.path.exists(text_path)):
            os.mkdir(text_path)
        with open(os.path.join(text_path,f"{id}.txt"),"w") as w_f:
            w_f.write(pdf_text)
            w_f.close()
    
        
