from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

class IntentPredictor:
    def __init__(self, model_path='models/quantized/model.bin'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = DistilBertForSequenceClassification.from_pretrained(model_path).to(self.device)

    def predict_intent(self, eeg_features):
        input_text = f"alpha: {eeg_features[0]}, beta: {eeg_features[1]}, mu: {eeg_features[2]}, asymmetry: {eeg_features[3]}"
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        logits = outputs.logits
        predicted_class_id = logits.argmax().item()
        
        return predicted_class_id
