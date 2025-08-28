import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import re

class PhiModel:
    def __init__(self, model_id, query, system_msg):
        self.query = query
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.model.config.pad_token_id = self.tokenizer.eos_token_id
        self.system_msg = system_msg

    def generate_prompt(self, ocr_text):
        return (
            self.system_msg +
            "Find any matches to the words from the Query in the TEXT.\n\n"
            f"Query = {self.query}\n\n"
            "TEXT:\n" 
            f"{ocr_text}\n"
            "TEXT END"
        )
    
    import re

    def find_last_dict_text(text: str) -> str | None:
        """
        Find and return the last dictionary-like string from the input text.
        Returns the raw string.
        """
        pattern = r"\{.*?\}"
        matches = re.findall(pattern, text, re.DOTALL)
        return matches[-1] if matches else None

    def generate_response(self, ocr_text):
        prompt = self.generate_prompt(ocr_text)
        # Tokenize and run generation
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=8192,
                do_sample=False,
                temperature=0.7,
                top_p=0.9,
                eos_token_id=self.tokenizer.eos_token_id
            )

        # Decode and print result
        result = self.tokenizer.decode(output[0], skip_special_tokens=True)
        text_response = self.find_last_dict_text(result)
        # Try to parse as JSON
        try:
            json_response = json.loads(text_response)
            print(f"Parsed JSON response: {json_response}")
            return json_response
        except json.JSONDecodeError:
            # If not valid JSON, return the raw text
            return result
