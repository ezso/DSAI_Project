from openai import OpenAI
import os
import json

class GPT4oMini:
    def __init__(self, model_id, query, system_msg):
        self.model_id = model_id
        self.query = query
        self.system_msg = system_msg
        self.api_key = os.getenv("OPENAI_API_KEY")

    def generate_user_input_text(self, ocr_text):
        return ("TASK:\n"
            f"Find any matches to the following words in the TEXT: \"{self.query}\".\n"
            "TASK END\n\n"
            "TEXT:\n" 
            f"{ocr_text}\n"
            "TEXT END"
        )
    
    def generate_response(self, ocr_text):
        """
        Special handling for the 4o-mini model.
        """
        user_text = self.generate_user_input_text(ocr_text)

        client = OpenAI(api_key = self.api_key)

        response = client.responses.create(
        model=self.model_id,
        input=[
            {
            "role": "system",
            "content": [
                {
                "type": "input_text",
                "text": self.system_msg
                }
            ]
            },
            {
            "role": "user",
            "content": [
                {
                "type": "input_text",
                "text": user_text
                }
            ]
            }
        ],
        text={
            "format": {
            "type": "text"
            }
        },
        reasoning={},
        tools=[],
        temperature=1,
        max_output_tokens=2048,
        top_p=1,
        store=True
        )

        text_response = response.output[0].content[0].text

        # Try to parse as JSON
        try:
            json_response = json.loads(text_response)
            print(f"Parsed JSON response: {json_response}")
            return json_response
        except json.JSONDecodeError:
            # If not valid JSON, return the raw text
            return text_response