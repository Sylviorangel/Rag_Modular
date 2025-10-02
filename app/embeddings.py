from typing import List, Sequence
from openai import OpenAI
import os

def _approx_tokens(text: str) -> int:
    # Aproximação simples (chars/4)
    return max(1, len(text) // 4)

class OpenAIEmbedder:
    def __init__(self, api_key: str, api_base: str | None, model: str):
        kwargs = {"api_key": api_key}
        if api_base:
            kwargs["base_url"] = api_base
        self.client = OpenAI(**kwargs)
        self.model = model

        # Configuráveis via .env
        self.max_req_tokens = int(os.getenv("EMBED_MAX_TOKENS_PER_REQ", "240000"))
        self.max_batch_size = int(os.getenv("EMBED_BATCH_SIZE", "128"))
        self.max_input_tokens = int(os.getenv("EMBED_MAX_INPUT_TOKENS", "7000"))

    def _flush(self, batch: List[str], out: List[List[float]]):
        if not batch:
            return
        resp = self.client.embeddings.create(model=self.model, input=batch)
        for d in resp.data:
            out.append(d.embedding)

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        seq = [t if isinstance(t, str) else str(t) for t in texts]

        out: List[List[float]] = []
        batch: List[str] = []
        tok_sum = 0

        for t in seq:
            t_tokens = _approx_tokens(t)

            # Trunca entradas individuais muito grandes
            if t_tokens > self.max_input_tokens:
                max_chars = self.max_input_tokens * 4
                t = t[:max_chars]
                t_tokens = _approx_tokens(t)

            if batch and (len(batch) + 1 > self.max_batch_size or tok_sum + t_tokens > self.max_req_tokens):
                self._flush(batch, out)
                batch = []
                tok_sum = 0

            batch.append(t)
            tok_sum += t_tokens

        self._flush(batch, out)
        return out
