import os
from tqdm import tqdm
import pdfplumber  # 用於從PDF文件中提取文字的工具
import json
import fitz
import pytesseract
from PIL import Image
# 載入參考資料，返回一個字典，key為檔案名稱，value為PDF檔內容的文本    
def load_data(source_path,save_name):
    save_path=os.path.join("../dataset/preliminary",save_name)
    corpus_dict=None
    if(os.path.exists(save_path)):
        with open(save_path,"r") as file:
            corpus_dict=json.load(file)
            file.close()
        return corpus_dict
    
    masked_file_ls = os.listdir(source_path)  # 獲取資料夾中的檔案列表
    corpus_dict = {int(file.replace('.pdf', '')): read_pdf(os.path.join(source_path, file)) for file in tqdm(masked_file_ls)}  # 讀取每個PDF文件的文本，並以檔案名作為鍵，文本內容作為值存入字典
    with open(save_path,"w") as save_file:
        json.dump(corpus_dict,save_file,ensure_ascii=False,indent=4)
        save_file.close()
    return corpus_dict

# 讀取單個PDF文件並返回其文本內容
def read_pdf(pdf_loc, page_infos: list = None):
    pdf = pdfplumber.open(pdf_loc)  # 打開指定的PDF文件

    # TODO: 可自行用其他方法讀入資料，或是對pdf中多模態資料（表格,圖片等）進行處理

    # 如果指定了頁面範圍，則只提取該範圍的頁面，否則提取所有頁面
    pages = pdf.pages[page_infos[0]:page_infos[1]] if page_infos else pdf.pages
    pdf_text = ''
    for _, page in enumerate(pages):  # 迴圈遍歷每一頁
        text = page.extract_text()  # 提取頁面的文本內容
        if text:
            pdf_text += text
    pdf.close()  # 關閉PDF文件


    if(not pdf_text): ## scanned pdf
        doc=fitz.open(pdf_loc)
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
            scanned_text = pytesseract.image_to_string(img,lang='chi_tra+eng')
            pdf_text+=scanned_text + "\n"
    return pdf_text  # 返回萃取出的文本