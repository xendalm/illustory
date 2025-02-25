import pdfplumber
import os

def extract_text_and_images(pdf_path, output_folder="output"):
    os.makedirs(output_folder, exist_ok=True)
    text_data = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages):
            gap = 50
            bbox = (0, gap, page.width, page.height - gap)
            cropped_page = page.within_bbox(bbox)
            text = cropped_page.extract_text()
            text_data.append(text)

            images = page.images
            for img_index, img in enumerate(images):
                img_data = img["stream"].get_data()
                img_ext = img["ext"] if "ext" in img else "jpg"
                img_filename = f"{output_folder}/page_{page_number + 1}_img_{img_index + 1}.{img_ext}"
                with open(img_filename, "wb") as img_file:
                    img_file.write(img_data)

    return text_data