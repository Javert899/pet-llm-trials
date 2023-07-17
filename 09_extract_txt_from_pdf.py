import os
import string
from PyPDF2 import PdfReader
import traceback


# Directory containing the PDF files
pdf_dir = 'pdf'

# Directory to save the text files
txt_dir = 'txt'

# Make sure output directory exists
os.makedirs(txt_dir, exist_ok=True)

# Function to check if majority of characters are printable
def is_text_readable(text, threshold=0.85):
    if len(text) == 0:
        return False
    num_printable_chars = sum(c in string.printable for c in text)
    return num_printable_chars / len(text) > threshold

# Iterate over all files in the PDF directory
for filename in os.listdir(pdf_dir):
    try:
        if filename.endswith('.pdf'):
            print(filename)
            # Open the PDF file
            with open(os.path.join(pdf_dir, filename), 'rb') as pdf_file:
                reader = PdfReader(pdf_file)

                # Initialize a string to store the text
                text = ''

                # Iterate over all pages in the PDF file
                for page in reader.pages:
                    extracted_text = page.extract_text()
                    if is_text_readable(extracted_text):
                        text += extracted_text

                # Create a corresponding text file and write the text into it
                if text:
                    text = text.lower()
                    if "process mining" in text:
                        with open(os.path.join(txt_dir, filename.replace('.pdf', '.txt')), 'w', encoding='utf8') as txt_file:
                            txt_file.write(text)
                    else:
                        print(filename+" extraction unsuccessful!!!!!")
                else:
                    print(filename+" with no text!!")
    except:
        traceback.print_exc()
