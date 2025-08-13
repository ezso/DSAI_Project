from dataset_collector import DatasetCollector
from ddb_api_calls import DDBAPI
from krakenOCR_model import ocr2text, xml2text
from regex_search_model import RegexSearchModel
from phi_models import PhiModel
from gpt_4o_mini import GPT4oMini
import os
import shutil
import warnings



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
def main():
    terms = ["Anarchismus", "Kommunismus", "Sozialismus", "Revolution"]
    # Define the model ID and system message
    # Uncomment the model_id you want to use
    # model_id = "microsoft/phi-1_5"
    # model_id = "microsoft/phi-3-mini-4k-instruct"
    # model_id = "gpt-4o-mini"
    model_id = "regex_search_model"  # Placeholder for regex search model

    # Set to True if you want to use Kraken OCR
    use_kraken = False

    system_msg = (
        "SYSTEM MESSAGE:\n\n"
        "The texts provided in the prompt are taken from OCR outputs of old newspapers. "
        "Some characters in the OCR text may be misread (e.g., 'h' as 'n', 'd' as 'o', 's' as 'ſ', 'i' as 't', 'l' as 't'), "
        "so you should be able to recognize and process these kinds of mistakes correctly. "
        "You should also consider different cognates of a keyword as matched. For example, several word forms, "
        "derivatives, and related terms of the German word 'Terrorismus' (terrorism): Terrorist, Terroranschlag, "
        "Terrororganisation, Antiterrorkampf, terrorisieren, etc.\n\n"
        "Give your answer in JSON format as follows:\n"
        "{\n"
        '  "success": boolean,\n'
        '  "word1": [],\n'
        '  "word2": [],\n'
        '  ...\n'
        '  "wordn": []\n'
        "}\n\n"
        "Example:\n"
        "If the query is {word1: \"Regelmäßig\", word2: \"Polizei\", word3: \"Terrorismus\"},\n"
        "the output JSON could be:\n"
        "{\n"
        '  "success": true,\n'
        '  "word1": ["Unregelmäßig"],\n'
        '  "word2": ["Polizisten", "Polizeiwagen"],\n'
        '  "word3": []\n'
        "}\n\n"
        "or\n\n"
        "{\n"
        '  "success": false,\n'
        '  "word1": [],\n'
        '  "word2": [],\n'
        '  "word3": []\n'
        "}\n\n"
        "depending on the matches found.\n\n"
        "SYSTEM MESSAGE END"
    )

    query = "{" + ", ".join([f'word{i+1}: "{term}"' for i, term in enumerate(terms)]) + "}"  # 'Anarchismus, Kommunismus, Sozialismus, Revolution'
    
    # Load tokenizer and model
    if model_id == "regex_search_model":
        model = RegexSearchModel(terms)
    elif model_id == "gpt-4o-mini":
        model = GPT4oMini(model_id, query=query, system_msg=system_msg)
    elif model_id == "microsoft/phi-3-mini-4k-instruct":
        model = PhiModel(model_id, query=query, system_msg=system_msg)

    # Initialize DDB API with parameters
    # The first parameter is the number of items to fetch, the second is the offset
    rows = 10
    offset = 0
    ddb = DDBAPI(rows, offset)
    # get the ids from ddb 
    response = ddb.get_ddb_data()
    item_ids = ddb.get_ids(response)
    # Create a csv file to store the results
    csv_file = f"dataset_{model_id}_{rows}_{offset}.csv"
    collector = DatasetCollector(csv_file, terms)
    for item_id in item_ids:
        if ddb.in_visited_ids(item_id):
            print(f"Skipping already visited ID: {item_id}")
            continue
        ddb.add_id_to_visited(item_id)

        if use_kraken:
            # get the xml and image files
            folder, numpages, issued, publisher = ddb.get_img_urls_and_xml(item_id)
        else:
            folder, numpages, issued, publisher = ddb.get_xmls_only(item_id)
        
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
            
            if use_kraken:
                page_str = f"{page:02d}"
                image_path = os.path.join(folder, f"page_{page_str}.jpg")

                if not os.path.exists(image_path):
                    print(f"Skipping missing files for item {item_id}, page {page_str}")
                    continue
                # Perform OCR using Kraken
                print(f"Processing item {item_id}, page {page_str}")
                warnings.filterwarnings("ignore", message="Using legacy polygon extractor, as the model was not trained with the new method.")
                full_text = ocr2text(image_path, xml_path)
            else:
                full_text = xml2text(xml_path)
            # start processing the text with LLM by chunking it into smaller parts
            # to avoid exceeding the token limit
            max_chunk_length = 100 

            for idx, chunk in enumerate(chunk_text_by_words(full_text, max_chunk_length), 1):
                print(f"Processing item {item_id}, chunk {idx}")
                model_response = model.generate_response(chunk)
                # If model_response is not a dict (e.g., JSON decode failed), wrap it
                if not isinstance(model_response, dict):
                    model_response = {"llm_response": str(model_response)}
                collector.add_row({
                    "item_id": item_id,
                    "publisher": publisher,
                    "pub_date": issued,
                    "page_num": page,
                    "chunk": chunk,
                    **model_response  # if model_response is a dict
                })
        
        # delete the image and xml files
        clean_the_folder()

if __name__ == "__main__":
    main()