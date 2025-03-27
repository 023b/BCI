# ollama_sender.py - Send state to LLM (Ollama on phone) and get response

import requests

class OllamaSender:
    def __init__(self, model='mistral', host='http://localhost:11434'):
        self.url = f"{host}/api/chat"
        self.model = model

    def send(self, fused_state):
        prompt = self.build_prompt(fused_state)
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are an emotionally aware AI assistant."},
                {"role": "user", "content": prompt}
            ]
        }

        try:
            response = requests.post(self.url, json=payload)
            response.raise_for_status()
            data = response.json()
            return data['message']['content'] if 'message' in data else "[No Response]"
        except Exception as e:
            return f"[Ollama Error] {e}"

    def build_prompt(self, state):
        intent = state.get("intent", "unknown")
        engagement = state.get("engagement", 0.0)
        confidence = state.get("confidence", 0.0)
        return (
            f"The user is showing intent: '{intent}', engagement: {engagement:.2f}, confidence: {confidence:.2f}.\n"
            f"Respond appropriately based on this information."
        )
