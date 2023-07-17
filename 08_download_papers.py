import requests
import os
import time
import traceback


# List of PDF urls
pdf_urls = ['https://www.vdaalst.com/publications/p'+str(i+1)+'.pdf' for i in range(1335)]
pdf_urls.reverse()

# Directory to save the PDF files
output_dir = 'pdf'

# Make sure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Iterate over each url
for url in pdf_urls:
    time.sleep(0.2)
    print("\n\ndownloading: "+url)

    try:
        response = requests.get(url)

        # Check if the request was successful
        if response.status_code == 200:
            # Split the url by '/' and take the last element as the file name
            file_name = url.split('/')[-1]

            # Open a file in the specified directory and write the content into it
            with open(os.path.join(output_dir, file_name), 'wb') as file:
                file.write(response.content)
    except:
        traceback.print_exc()
