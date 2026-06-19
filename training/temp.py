from train import get_tokenizer

tok = get_tokenizer("roberta-base")

print(type(tok))
print(tok.backend_tokenizer.model)
print(tok.vocab_files_names)
