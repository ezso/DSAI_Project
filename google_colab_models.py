from google.colab import ai
import json

class GoogleColabModel:
    def __init__(self, model_id, query, system_msg):
        self.model_id = model_id
        self.query = query
        self.system_msg = system_msg

    def generate_prompt(self, ocr_text):
        return (
            self.system_msg + "\n\n" +
            "Find any matches to the words from the Query in the TEXT.\n\n"
            f"Query = {self.query}\n\n"
            "TEXT:\n" 
            f"{ocr_text}\n"
            "TEXT END"
        )
    
    def generate_response(self, ocr_text):
        prompt = self.generate_prompt(ocr_text)
        response = ai.generate_text(prompt, model_name=self.model_id)

        # Try to parse as JSON
        try:
            json_response = json.loads(response)
            print(f"Parsed JSON response: {json_response}")
            return json_response
        except json.JSONDecodeError:
            # If not valid JSON, return the raw text
            return response