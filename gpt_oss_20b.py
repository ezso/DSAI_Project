import json

class GPTOSS_20bModel:
    def __init__(self, pipe, query, system_msg):
        """
        model_id: Hugging Face model ID (e.g., 'openai/gpt-oss-20b')
        query: list of words or terms (optional, used if you want search tasks)
        system_msg: system instruction for the assistant (optional)
        """
        self.query = query
        self.system_msg = system_msg

        self.pipe = pipe

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

        outputs = self.pipe(
            messages,
            max_new_tokens=4096,
        )

        content = outputs[0]["generated_text"][-1]["content"]
        flag="assistantfinal"
        final_answer = content.split(flag, 1)[-1].strip()
        try:
            return json.loads(final_answer)
        except Exception:
            return content
