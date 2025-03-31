import os
import requests

def download_hdri(url, save_path):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(1024):
                file.write(chunk)
        print(f"Downloaded: {save_path}")
    else:
        print(f"Failed to download: {url}")

def main():
    with open('img-url-1k.txt', 'r') as file:
        urls = file.readlines()

    save_directory = 'hdri_files'
    os.makedirs(save_directory, exist_ok=True)

    for url in urls:
        url = url.strip()
        if url:
            file_name = os.path.basename(url)
            save_path = os.path.join(save_directory, file_name)
            download_hdri(url, save_path)

if __name__ == "__main__":
    main()
