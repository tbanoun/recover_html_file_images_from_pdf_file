import cv2, re, shutil, os
import numpy as np
import base64
from io import BytesIO
from PIL import Image
from skimage.metrics import structural_similarity as compare_ssim
from bs4 import BeautifulSoup
from word_to_html import extract_images_from_pdf as pdf_img
# pip install opencv-python-headless numpy scikit-image Pillow


def base64_to_image(base64_string):
    """Convert image base64 to png img"""
    img_data = base64.b64decode(base64_string)
    img = Image.open(BytesIO(img_data))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def image_to_base64(image):
    """Convert image to base64"""
    _, buffer = cv2.imencode('.png', image)
    return base64.b64encode(buffer).decode('utf-8')

def compare_images(img1, img2):
    """
        Tester si deux images sont identique
    """
    image1 = base64_to_image(img1.split(",")[1])
    image2 = base64_to_image(img2.split(",")[1])
    # Convert images to grayscale
    img1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    # Resize images to the same size
    img1_gray = cv2.resize(img1_gray, (100, 100))
    img2_gray = cv2.resize(img2_gray, (100, 100))
    
    # Compute structural similarity index (SSIM)
    score, diff = compare_ssim(img1_gray, img2_gray, full=True)
    print("Scoore: ",score)
    return score

def extract_images(image_path, page):
    """
        recuperer les sous images d'une image ;)
    """
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray, 30, 200)
    
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    images = []
    i = 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 50 and h > 50:  # Filtrer les petits contours
            roi = image[y:y+h, x:x+w]
            images.append(roi)
            i+=1
            output_dir = f"images_Pdf"
            output_path = os.path.join(output_dir, f'sub_image_page{page}_{i}.png')
            
            # Enregistrer l'image
            print("image save", output_path)
            cv2.imwrite(output_path, roi)
    
    return images


def update_image_html_using_pdf(html_file_path, pdf_file_path):
    # mettre a jour le dossier qui va contenir les images du pdf 
    imgdir = "images_Pdf"
    if os.path.exists(imgdir):
        shutil.rmtree(imgdir)
        os.mkdir(imgdir)
    # lire le contenu du fichier html
    with open(html_file_path, 'r') as f:
        html_content = f.read()
        soup = BeautifulSoup(html_content, 'html.parser')
    # selectionner toutes les pages
    pages = soup.find_all('div', class_=re.compile(r'page\d+'))
    # parcourir les pages
    for index, page in enumerate(pages):
        images = page.find_all('img')
        if len(images)<=0: continue #si y a pas des images on passe a une autre page
        # recupeper les images depuis le pdf
        images_pdf = pdf_img(pdf_file_path, [index+1, index+2])
        if not images_pdf: continue
        # sauvgarder l'image dans le dossier images_pdf
        chemin_img = f"{imgdir}/image_page{index}.png"
        try: images_pdf[0].save(chemin_img, format="PNG")
        except Exception: continue
        # recuperper les sous images de l'images de pdf
        images_child = extract_images(chemin_img, index)
        if not images_child: continue
        if len(images_child) != len(images): continue
        for i, img in enumerate(images): #on parcour les images si y a une resemblance avec le pdf on les remplace sionon on continue
            try: 
                imgPdf64 = f"data:image/png;base64,{image_to_base64(images_child[i])}"
            except Exception: continue
            if index == 28: print(" \n\n  Hello world  \n\n")
            compar_img = compare_images(imgPdf64, img['src'])
            if not compar_img: continue
            img64 = img
            try: img64['src'] = f"data:image/png;base64,{image_to_base64(images_child[i])}"
            except Exception: continue
            img.replace_with(img64)
    with open("index33.html", "w") as html_file:
        html_file.write(soup.prettify(formatter=None))

update_image_html_using_pdf("Parite5_word_OPTIMISED.html", "Parite5/2023 04 01 TSCA URD 2022 VDEF2.pdf")