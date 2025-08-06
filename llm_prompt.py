import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI
import os

class LLMPrompt:
    def __init__(self, model_id, query, system_msg):
        self.query = query
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
        self.model.config.pad_token_id = self.tokenizer.eos_token_id
        self.system_msg = system_msg
        self.api_key = os.getenv("OPENAI_API_KEY")

    def generate_prompt_phi(self, ocr_text):
        prompt = (
            self.system_msg +
            f"OCR Text:\n\"\"\"{ocr_text}\"\"\"\n\n"
            f"Search Query:\n\"\"\"{self.query}\"\"\"\n\n"
            "Are there any matches to the search query? If yes then respond with \"Found: <word1, word2, ...>.\" "
            "Otherwise just respond with \"Not found.\""
        )
        return prompt


    def generate_phi_response(self, ocr_text):
        prompt = self.generate_prompt_phi(ocr_text)
        # Tokenize and run generation
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=False,
                temperature=0.7,
                top_p=0.9,
                eos_token_id=self.tokenizer.eos_token_id
            )

        # Decode and print result
        result = self.tokenizer.decode(output[0], skip_special_tokens=True)
        clean_prompt = prompt.replace("<|user|>", "").replace("<|assistant|>", "").strip()
        return result.replace(clean_prompt, "", 1).strip()
    
    def generate_user_input_text(self, ocr_text):
        return ("TASK:\n"
            f"Find any matches to the following words in the TEXT: \"{self.query}\".\n"
            "If success then respond with \"Found: <word1, word2, ...>.\" Otherwise just respond with \"Not found.\"\n"
            "TASK END\n\n"
            "TEXT:\n" 
            f"{ocr_text}\n"
            "TEXT END"
        )

    def generate_4o_mini_response(self, ocr_text):
        """
        Special handling for the 4o-mini model.
        """
        user_text = self.generate_user_input_text(ocr_text)

        client = OpenAI(api_key = self.api_key)

        response = client.responses.create(
        model="gpt-4o-mini",
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

        return response.output[0].content[0].text