from dataset_collector import DatasetCollector
from ddb_api_calls import DDBAPI
from krakenOCR import ocr2text, xml2text
from llm_prompt import LLMPrompt
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
    query = "Anarchismus, Terrorismus, Revolution, Imperialismus, Sozialismus, Kommunist"
    # Load model and tokenizer
    # model_id = "microsoft/phi-3-mini-4k-instruct"
    # model_id = "microsoft/phi-1_5"
    model_id = "microsoft/phi-3-mini-4k-instruct"

    system_msg = (
        "The texts provided in the prompt are taken from OCR outputs from old german newspapers. "
        "Some characters in the OCR text may be misread (e.g. 'h' as 'n', 'd' as 'o', 's' as 'ſ', "
        "'i' as 't', 'l' as 't'), so you should be able to distinguish these kinds of mistakes and process them correctly. "
        "You should also consider different cognates of a key word as matched. "
        "For example: several word forms, derivatives, and related terms of the German word 'Terrorismus' (terrorism): "
        "Terrorist, Terroranschlag, Terrororganisation, Antiterrorkampf, terrorisieren, etc.\n\n"
    )
    # Load tokenizer and model
    llm = LLMPrompt(model_id, query, system_msg)

    # Set to True if you want to use Kraken OCR
    use_kraken = False 
    # Set to True if you want to use GPT-4o-mini
    use_4o_mini = True
    # Initialize DDB API with parameters
    # The first parameter is the number of items to fetch, the second is the offset
    rows = 10
    offset = 0
    ddb = DDBAPI(rows, offset)
    # get the ids from ddb 
    response = ddb.get_ddb_data()
    item_ids = ddb.get_ids(response)
    # Create a csv file to store the results
    csv_file = f"dataset_{rows}_{offset}.csv"
    collector = DatasetCollector(csv_file)
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
                if use_4o_mini:
                    # Generate response using GPT-4o-mini
                    llm_response = llm.generate_4o_mini_response(chunk)
                else:
                    # Generate response using phi-3-mini-4k-instruct
                    llm_response = llm.generate_phi_response(chunk)
                collector.add_row(
                    item_id=item_id,
                    publisher=publisher,
                    pub_date=issued,
                    page_num=page,
                    chunk=chunk,
                    response=llm_response
                )
        
        # delete the image and xml files
        clean_the_folder()

if __name__ == "__main__":
    main()