import torch
from langchain.embeddings.base import Embeddings
from typing import List
import torch.nn.functional as F
from torch import Tensor


class CustomEmbeddings(Embeddings):
    def __init__(self, embedder, tokenizer, is_hf_model=False, token_pool=True):
        super().__init__()

        self.tokenizer = tokenizer
        self.embedder = embedder
        self.is_hf_model = is_hf_model
        self.token_pool = token_pool


    def last_token_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]

            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        
        for text in texts:
            embedding = None
            #   If tokenizer is none, we're using a huggingface embedder so handle embedding differently.
            # if self.tokenizer is not None:
            if self.is_hf_model:
                text = self.tokenizer.encode(text, bos=False, eos=False)
                text = torch.tensor(text, dtype=torch.int).cuda()
                embedding = self.embedder(torch.tensor(text, dtype=torch.int).cuda())
            elif self.token_pool:
                text = [text]
                max_length = 4096# Tokenize the input texts
                batch_dict = self.tokenizer(text, max_length=max_length - 1, return_attention_mask=False, padding=False, truncation=True)# append eos_token_id to every input_ids
                batch_dict['input_ids'] = [input_ids + [self.tokenizer.eos_token_id] for input_ids in batch_dict['input_ids']]
                batch_dict = self.tokenizer.pad(batch_dict, padding=True, return_attention_mask=True, return_tensors='pt').to("cuda:0")

                outputs = self.embedder(**batch_dict)
                embedding = self.last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

                # normalize embeddings
                embedding = F.normalize(embedding, p=2, dim=1)[0].tolist()
            else:
                text = [text]
                # embedding = self.embedder(text, return_tensors="pt")["input_ids"].to("cuda")
                # embedding = embedding.type(torch.float)

                text = self.tokenizer(text, return_tensors="pt").to("cuda:0")#["input_ids"]#.to("cuda").type(torch.float)
                embedding = self.embedder(text["input_ids"])[0]

            # embedding = torch.mean(embedding, dim=0).tolist()
            # embedding = torch.max(embedding, dim=0)[0].tolist()

            embeddings.append(embedding)

        return embeddings


    def embed_query(self, text: str) -> List[float]:
        embedding = None
        # if self.tokenizer is not None:

        print(f"Query to embed:  {text}")
        if self.is_hf_model:
            text = self.tokenizer.encode(text, bos=False, eos=False)
            text = torch.tensor(text, dtype=torch.int).cuda()
            embedding = self.embedder(torch.tensor(text, dtype=torch.int).cuda())
        elif self.token_pool:
            text = [text]
            max_length = 4096# Tokenize the input texts
            batch_dict = self.tokenizer(text, max_length=max_length - 1, return_attention_mask=False, padding=False, truncation=True)# append eos_token_id to every input_ids
            batch_dict['input_ids'] = [input_ids + [self.tokenizer.eos_token_id] for input_ids in batch_dict['input_ids']]
            batch_dict = self.tokenizer.pad(batch_dict, padding=True, return_attention_mask=True, return_tensors='pt').to("cuda:0")

            outputs = self.embedder(**batch_dict)
            embedding = self.last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

            # normalize embeddings
            embedding = F.normalize(embedding, p=2, dim=1)[0].tolist()
        else:
            text = [text]
            # embedding = self.embedder(text, return_tensors="pt")["input_ids"].to("cuda")
            # embedding = embedding.type(torch.float)

            text = self.tokenizer(text, return_tensors="pt").to("cuda:0")
            embedding = self.embedder(text["input_ids"])[0]

        # embedding = torch.mean(embedding, dim=0).tolist()
        # embedding = torch.max(embedding, dim=0)[0].tolist()

        return embedding