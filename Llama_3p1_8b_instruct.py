import json

class Llama3p1_8bModel:
    def __init__(self, pipe, query, system_msg):
        self.query = query
        self.system_msg = system_msg

        self.pipeline = pipe

    def generate_user_input_text(self, ocr_text):
        return (
            "Find any matches to the words from the Query in the TEXT.\n\n"
            f"Query = {self.query}\n\n"
            "TEXT:\n" 
            f"{ocr_text}\n"
            "TEXT END"
        )

    def generate_response(self, ocr_text):
        user_text = self.generate_user_input_text(ocr_text)

        messages = [
            {"role": "system", "content": self.system_msg},
            {"role": "user", "content": user_text},
        ]

        outputs = self.pipeline(
            messages,
            max_new_tokens=256
        )

        # HuggingFace chat pipeline returns a list of dicts with "generated_text"
        text_response = outputs[0]["generated_text"][-1]['content']

        # Try JSON parsing
        try:
            return json.loads(text_response)
        except Exception:
            return text_response
