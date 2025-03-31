import os
import requests

url_list_file = '/materials/download_1k.txt'
download_dir = '/materials'

os.makedirs(download_dir, exist_ok=True)

with open(url_list_file, 'r') as file:
    urls = file.readlines()

for url in urls:
    url = url.strip()
    if url:  
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            filename = os.path.join(download_dir, os.path.basename(url))
            with open(filename, 'wb') as out_file:
                for chunk in response.iter_content(chunk_size=8192):
                    out_file.write(chunk)
            print(f"Downloaded: {url}")
        except requests.exceptions.RequestException as e:
            print(f"Failed to download {url}: {e}")

print("All downloads completed.")
