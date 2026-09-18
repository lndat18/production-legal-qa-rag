from dotenv import load_dotenv
load_dotenv()

import litserve as ls
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "AITeamVN/Vietnamese_Reranker"
MAX_LENGTH = 2304  # 256 query + 2048 passage, theo model card

class RerankerAPI(ls.LitAPI):
    def setup(self, device):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        self.model.to(device).eval()
        self.device = device

    def decode_request(self, request):
        return request["query"], request["passages"]

    def predict(self, inputs):
        query, passages = inputs
        if not passages:
            return []
        pairs = [[query, p] for p in passages]
        with torch.no_grad():
            tokenized = self.tokenizer(
                pairs, padding=True, truncation=True,
                max_length=MAX_LENGTH, return_tensors="pt",
            ).to(self.device)
            scores = self.model(**tokenized, return_dict=True).logits.view(-1).float()
        return scores.tolist()

    def encode_response(self, scores):
        return {"scores": scores}

if __name__ == "__main__":
    server = ls.LitServer(RerankerAPI(), accelerator="auto")
    server.run(port=8000)
