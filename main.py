from dataset_collector import DatasetCollector
from ddb_api_calls import DDBAPI
import xml.etree.ElementTree as ET
import os
import shutil


def xml2text(xml_path):
    # ALTO XML uses a default namespace
    ns = {'alto': 'http://www.loc.gov/standards/alto/ns-v3#'}

    # Parse the XML file
    tree = ET.parse(xml_path)
    root = tree.getroot()

    lines = []

    # Traverse each TextLine in the XML
    for text_line in root.findall('.//alto:TextLine', ns):
        words = [string.attrib.get("CONTENT", "") for string in text_line.findall("alto:String", ns)]
        line_text = " ".join(words)
        if line_text.strip():
            lines.append(line_text)

    full_text = "\n".join(lines)

    return full_text

# This script fetches data from the DDB API, processes images and XML files, performs OCR, and interacts with a language model.

# Function to clean the folder by deleting all files and subfolders
def clean_the_folder(base_dir="ddb"):
    """
    Deletes all files and subfolders in the given base_dir.
    """

    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
        print(f"Deleted all contents in '{base_dir}'")
    else:
        print(f"Directory '{base_dir}' does not exist.")

# Function to chunk text into smaller parts for LLM processing
def chunk_text_by_words(text, max_words):
    words = text.split()
    for i in range(0, len(words), max_words):
        yield ' '.join(words[i:i + max_words])

# Main function to orchestrate the process
def main(terms, zdb, rows, offset, model_id, system_msg, pipeline=None):

    query = "{" + ", ".join([f'word{i+1}: "{term}"' for i, term in enumerate(terms)]) + "}"  # 'Anarchismus, Kommunismus, Sozialismus, Revolution'
    
    # Load tokenizer and model
    if model_id == "regex_search_model":
        from regex_search_model import RegexSearchModel
        model = RegexSearchModel(terms)
    elif model_id == "gpt-4o-mini":
        from gpt_4o_mini import GPT4oMini
        model = GPT4oMini(model_id, query=query, system_msg=system_msg)
    elif model_id == "phi-3-mini-4k-instruct":
        from phi_models import PhiModel
        model = PhiModel("microsoft/phi-3-mini-4k-instruct", query=query, system_msg=system_msg)
    elif model_id == "Llama-3.1-8b-instruct":
        from Llama_3p1_8b_instruct import Llama3p1_8bModel
        model = Llama3p1_8bModel(pipeline, query=query, system_msg=system_msg)
    elif model_id == "gpt-oss-20b":
        from gpt_oss_20b import GPTOSS_20bModel
        model = GPTOSS_20bModel(pipeline, query=query, system_msg=system_msg)
    elif model_id == "google/gemini-2.5-flash-lite":
        from google_colab_models import GoogleColabModel
        model = GoogleColabModel(model_id, query=query, system_msg=system_msg)

    # Initialize DDB API with parameters
    # The first parameter is the number of items to fetch, the second is the offset

    ddb = DDBAPI(zdb, rows, offset)
    # get the ids from ddb 
    response = ddb.get_ddb_data()
    item_ids = ddb.get_ids(response)
    # Create a csv file to store the results
    model_name = model_id.replace("/", "-")
    csv_file = f"dataset_{model_name}_{rows}_{offset}.csv"
    collector = DatasetCollector(csv_file, terms)
    for item_id in item_ids:
        if ddb.in_visited_ids(item_id):
            print(f"Skipping already visited ID: {item_id}")
            continue
        ddb.add_id_to_visited(item_id)

        folder, numpages, issued, publisher, title = ddb.get_xmls_only(item_id)
        
        if folder == None or numpages == 0:
            print(f"Skipping item {item_id} due to a download error.")
            continue

        # use ocr to extract text from the image
        # and return write the extracted text to a file in single line
        for page in range(1, numpages + 1):
            xml_path = os.path.join(folder, f"page_{page}.xml")

            if not os.path.exists(xml_path):
                print(f"Skipping missing files for item {item_id}, page {page}")
                continue

            full_text = xml2text(xml_path)
            # start processing the text with LLM by chunking it into smaller parts
            # to avoid exceeding the token limit
            max_chunk_length = 100 

            for idx, chunk in enumerate(chunk_text_by_words(full_text, max_chunk_length), 1):
                print(f"Processing item {item_id}, chunk {idx}")
                model_response = model.generate_response(chunk)
                # If model_response is not a dict (e.g., JSON decode failed), wrap it
                if not isinstance(model_response, dict):
                    model_response = {
                        "success": False, 
                        **{term: [] for term in terms},
                        "llm_response": str(model_response)
                    }
                collector.add_row({
                    "item_id": item_id,
                    "publisher": publisher,
                    "title": title,
                    "pub_date": issued,
                    "page_num": page,
                    "chunk": chunk,
                    **model_response  # if model_response is a dict
                })
        
        # delete the image and xml files
        clean_the_folder()
