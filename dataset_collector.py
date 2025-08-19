# **** |id|date|page number|chunk|response| ****
import csv
import os

class DatasetCollector:
    def __init__(self, csv_file, terms):
        self.csv_file = csv_file
        # Create the CSV file and write the header if it doesn't exist
        if not os.path.exists(csv_file):
            with open(csv_file, mode='w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["item_id", "publisher", "pub_date", "page_num", "chunk", "success"] + terms + ["llm_response"])

    def add_row(self, row_dict):
        with open(self.csv_file, mode='a', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row_dict.keys())
            writer.writerow(row_dict)