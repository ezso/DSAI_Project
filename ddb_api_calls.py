import requests
from urllib.parse import urlparse
import os
import time
import re

class DDBAPI:
    def __init__(self, rows, start):
        self.api_key = "OYSi9Dygc0XZ0Nvq2vgPxe4oXNmomCtWWZHM7CVd3Fo7iC0qKge1748029090188"  
        self.headers = {
            "Authorization": f'OAuth oauth_consumer_key="{self.api_key}"',
            "Accept": "application/json",
        }

        self.params = {
            "q": "*:*",
            "fq": [
                "publication_date:[1869-12-31T23:59:59.999Z TO 1940-12-31T23:59:59.999Z]"
            ],
            "rows": rows,
            "start": start
        }
        self.url = "https://api.deutsche-digitale-bibliothek.de/search/index/newspaper-issues/select"
        self.visited_ids = set()

    def get_ddb_data(self):
        """
        Fetches newspaper issue IDs from the Deutsche Digitale Bibliothek API.
        Returns a list of item IDs.
        """
        response = requests.get(
            self.url,
            headers=self.headers,
            params=self.params
        )

        return response

    def get_ids(self, response):
        all_data = response.json()
        ids = [doc['id'] for doc in all_data['response']['docs']]
        return ids
    
    def add_id_to_visited(self, item_id):
        self.visited_ids.add(item_id)
    
    def in_visited_ids(self, item_id):
        """
        Checks if the item_id is already in the visited_ids set.
        """
        return item_id in self.visited_ids

    def get_xmls_only(self, item_id, base_dir="ddb"):
        item_url = f"https://api.deutsche-digitale-bibliothek.de/items/{item_id}"
        response = requests.get(item_url, headers=self.headers)
        if response.status_code != 200:
            print(f"❌ Failed to fetch item {item_id}")
            return None, 0, '', ''

        try:
            data = response.json()
            mets_xml = data["source"]["record"]["$"]
            issued = data['edm']['RDF']['ProvidedCHO']['issued']
            publisher = data['edm']['RDF']['ProvidedCHO']['publisher']['$']
            xml_links = re.findall(r'https://[^\s"]+\.xml', mets_xml)
            xml_links = [url for url in xml_links if url.startswith("https://api.deutsche-digitale-bibliothek.de/binary/")]

            # Create output folder
            folder = os.path.join(base_dir, item_id)
            os.makedirs(folder, exist_ok=True)
            print(f"\n📥 Downloading XMLs to: {folder}")

            for i, xml_url in enumerate(xml_links, 1):
                xml_resp = requests.get(xml_url)
                if xml_resp.status_code == 200:
                    xml_path = os.path.join(folder, f"page_{i}.xml")
                    with open(xml_path, "wb") as f:
                        f.write(xml_resp.content)
                    print(f"✔ Saved {xml_path}")
                else:
                    print(f"❌ Failed to download XML: {xml_url}")

            return folder, len(xml_links), issued, publisher

        except Exception as e:
            print(f"❗ Error with item {item_id}: {e}")
            return None, 0, '', ''

    def get_img_urls_and_xml(self, item_id, base_dir="ddb"):

        # Get item metadata
        item_url = f"https://api.deutsche-digitale-bibliothek.de/items/{item_id}"
        response = requests.get(item_url, headers=self.headers)
        if response.status_code != 200:
            print(f"Failed to fetch item {item_id}")
            return None, 0, '', ''

        try:
            data = response.json()

            # Get the image URL
            img_url = data["edm"]["RDF"]["Aggregation"]["isShownBy"]["@resource"]
            issued = data['edm']['RDF']['ProvidedCHO']['issued']
            publisher = data['edm']['RDF']['ProvidedCHO']['publisher']['$']
            parsed_url = urlparse(img_url)
            basename = os.path.basename(parsed_url.path)      # e.g., MB12035_01.jpg
            basecode = basename.split("_")[0]                 # e.g., MB12035
            folder = os.path.join(base_dir, basecode)
            os.makedirs(folder, exist_ok=True)

            print(f"\n📥 Downloading images to: {folder}")
            # Download images
            for page in range(1, 100):  # Max 100 pages
                page_str = f"{page:02d}"
                page_url = f"{parsed_url.scheme}://{parsed_url.netloc}{os.path.dirname(parsed_url.path)}/{basecode}_{page_str}.jpg"
                head = requests.head(page_url)
                if head.status_code == 200:
                    img_data = requests.get(page_url).content
                    img_path = os.path.join(folder, f"page_{page_str}.jpg")
                    with open(img_path, "wb") as f:
                        f.write(img_data)
                    print(f"✔ Saved {img_path}")
                    time.sleep(0.5)
                else:
                    print(f"❌ No more images after page {page_str}")
                    break

            # Parse METS and download page XMLs
            print(f"\n📥 Downloading XMLs to: {folder}")
            mets_xml = data["source"]["record"]["$"]
            xml_links = re.findall(r'https://[^\s"]+\.xml', mets_xml)
            # Filter only DDB binary links
            xml_links = [url for url in xml_links if url.startswith("https://api.deutsche-digitale-bibliothek.de/binary/")]

            for i, xml_url in enumerate(xml_links, 1):
                xml_resp = requests.get(xml_url)
                if xml_resp.status_code == 200:
                    xml_path = os.path.join(folder, f"page_{i}.xml")
                    with open(xml_path, "wb") as f:
                        f.write(xml_resp.content)
                    print(f"✔ Saved {xml_path}")
                else:
                    print(f"❌ Failed to download XML: {xml_url}")

        except Exception as e:
            print(f"❗ Error with item {item_id}: {e}")
            return None, 0, '', ''
        
        return folder, len(xml_links), issued, publisher
