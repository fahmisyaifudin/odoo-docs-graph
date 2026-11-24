from mlx_embeddings.utils import load

DEFAULT_MLX_MODEL = "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ"


class MLXEmbedding:
    def __init__(self, model: str = DEFAULT_MLX_MODEL):
        self.model_name = model
        self.model, self.tokenizer = load(model)

    def document_embedding(self, document: str) -> list[float]:
        input_ids = self.tokenizer.encode(document, return_tensors="mlx")
        outputs = self.model(input_ids)
        return outputs.text_embeds.tolist()[0]  # mean pooled and normalized
