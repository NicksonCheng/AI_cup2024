import re
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

# 讀取單個PDF文件並返回其中文文本內容
def read_pdf(pdf_loc, page_infos: list = None, max_len=256, overlap_len=100):
    pdf = pdfplumber.open(pdf_loc)  # 打開指定的PDF文件

    # 如果指定了頁面範圍，則只提取該範圍的頁面，否則提取所有頁面
    pages = pdf.pages[page_infos[0]:page_infos[1]] if page_infos else pdf.pages
    page_content = []

    # 提取並清理每一頁的中文文本內容
    for _, page in enumerate(pages):  # 迴圈遍歷每一頁
        text = page.extract_text()  # 提取頁面的文本內容
        if text:
            # 過濾掉非中文字符
            chinese_text = re.sub(r'[^\u4e00-\u9fff]', '', text)
            if len(chinese_text) > 10:
                page_content.append(chinese_text)
    pdf.close()  # 關閉PDF文件

    # 如果沒有提取到文本（處理掃描版PDF）
    if not page_content:
        doc = fitz.open(pdf_loc)
        zoom_x = 2.0  
        zoom_y = 2.0  
        mat = fitz.Matrix(zoom_x, zoom_y)  
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(matrix=mat)  # Convert page to an image

            # Convert pixmap to PIL Image for OCR
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            # Extract text using OCR
            scanned_text = pytesseract.image_to_string(img, lang='chi_tra+eng')
            # 過濾掉非中文字符
            chinese_text = re.sub(r'[^\u4e00-\u9fff]', '', scanned_text)
            if len(chinese_text) > 10:
                page_content.append(chinese_text)

    # 合併所有中文文本為一個長字符串並進行滑動窗口分割
    all_str = ''.join(page_content)
    cleaned_chunks = []
    i = 0
    while i < len(all_str):
        cur_s = all_str[i:i + max_len]
        if len(cur_s) > 10:
            cleaned_chunks.append(cur_s)
        i += (max_len - overlap_len)

    return cleaned_chunks  # 返回清理後的中文文本片段