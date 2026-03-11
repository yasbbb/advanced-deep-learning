import abc

import torch


def load() -> torch.nn.Module:
    from pathlib import Path

    model_name = "AutoregressiveModel"
    model_path = Path(__file__).parent / f"{model_name}.pth"
    print(f"Loading {model_name} from {model_path}")
    return torch.load(model_path, weights_only=False)


class Autoregressive(abc.ABC):
    """
    Base class for all autoregressive models.
    Implement a specific model below.
    """

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Take a tensor x (B, h, w) if integers as input.
        Produce a probability over the next token as an output (B, h, w, n_token).
        Make sure the model is auto-regressive:
          - The first output result[:, 0, 0] does not depend on any input
          - The second output result[:, 0, 1] depends only on x[:, 0, 0]
          - etc.

        Hint 1: Flatten the tensor into a sequence.
        Hint 2: A positional embedding can help, but is not required.
        Hint 3: You need to shift the input sequence by 1 position. Do this after embedding the
                values, and before passing them through your model. (torch.concat or
                torch.nn.ConstantPad1d both work)
        """

    def generate(self, B: int = 1, h: int = 20, w: int = 30, device=None) -> torch.Tensor:  # noqa
        """
        Use your generative model to produce B new token images of size (B, h, w) and type (int/long).
        """


class AutoregressiveModel(torch.nn.Module, Autoregressive):
    """
    Implement an auto-regressive model.
    The input is a set of patch tokens (integers), the output is an image of probability.
    You need to implicitly shift your inputs by one position in the forward pass.
    Make sure n_tokens matches your BSQ dimension (2**codebook_bits_).

    Hint: You will need the torch.nn.Embedding function
    Hint: You can use torch.nn.TransformerEncoderLayer if you'd like
    Hint: You can complete this homework without using positional embeddings
    """

    def __init__(self, d_latent: int = 128, n_tokens: int = 2**10):
        super().__init__()
        self.d_latent = d_latent
        self.n_tokens = n_tokens
        self.max_seq_len = 1024  # enough for 20x30 or 30x20 token grids

        self.token_embedding = torch.nn.Embedding(n_tokens, d_latent)
        self.pos_embedding = torch.nn.Parameter(
            torch.zeros(1, self.max_seq_len, d_latent)
        )

        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_latent,
            nhead=8,
            dim_feedforward=4 * d_latent,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.output_proj = torch.nn.Linear(d_latent, n_tokens)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # x: (B, h, w) of integer token ids
        B, h, w = x.shape
        T = h * w

        if T > self.max_seq_len:
            raise ValueError(
                f"Sequence length {T} exceeds max_seq_len={self.max_seq_len}"
            )

        # Flatten spatial grid into sequence: (B, T)
        x_flat = x.reshape(B, T).long()

        # Embed tokens first
        emb = self.token_embedding(x_flat)  # (B, T, d_latent)

        # Shift by 1 position AFTER embedding:
        # first position is all zeros, later positions see previous embedded token
        zero_tok = torch.zeros(
            B, 1, self.d_latent, device=emb.device, dtype=emb.dtype
        )
        emb_shifted = torch.cat([zero_tok, emb[:, :-1, :]], dim=1)  # (B, T, d_latent)

        # Add positional embeddings
        emb_shifted = emb_shifted + self.pos_embedding[:, :T, :]

        # Causal attention mask: prevent attending to future tokens
        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )

        hidden = self.transformer(emb_shifted, mask=causal_mask)  # (B, T, d_latent)
        logits = self.output_proj(hidden)  # (B, T, n_tokens)

        # Reshape back to image grid of token logits
        logits = logits.reshape(B, h, w, self.n_tokens)

        return logits, {}

    def generate(self, B: int = 1, h: int = 30, w: int = 20, device=None) -> torch.Tensor:  # noqa
        if device is None:
            device = next(self.parameters()).device

        T = h * w
        if T > self.max_seq_len:
            raise ValueError(
                f"Sequence length {T} exceeds max_seq_len={self.max_seq_len}"
            )

        # Start with an empty token canvas
        tokens = torch.zeros((B, T), device=device, dtype=torch.long)

        for t in range(T):
            current = tokens.reshape(B, h, w)
            logits, _ = self.forward(current)  # (B, h, w, n_tokens)
            logits_flat = logits.reshape(B, T, self.n_tokens)

            # Get distribution for current position t
            next_logits = logits_flat[:, t, :]  # (B, n_tokens)
            probs = torch.softmax(next_logits, dim=-1)

            # Sample token
            next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
            tokens[:, t] = next_token

        return tokens.reshape(B, h, w)