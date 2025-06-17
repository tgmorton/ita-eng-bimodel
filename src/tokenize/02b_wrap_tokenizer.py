# 02b_wrap_tokenizer.py
import os
from transformers import LlamaTokenizer


def main():
    """
    Loads the raw bilingual sentencepiece models and saves them in the Hugging Face
    transformers format.
    """
    print("\n--- Running Script 02b: Wrapping Bilingual Tokenizers for Transformers ---")
    base_dir = os.path.expanduser('~/ita-eng-bimodel')
    tokenizer_dir = os.path.join(base_dir, 'tokenizer')
    print(f"Reading from and writing to: {tokenizer_dir}\n")

    bilingual_configs = [
        {'name': '10_25_it_eng'},
        {'name': '25_25_it_eng'},
        {'name': '50_25_it_eng'},
        {'name': '10_25_eng_it'},
        {'name': '25_25_eng_it'},
        {'name': '50_25_eng_it'},
    ]

    for config in bilingual_configs:
        name = config['name']
        print(f"----- Wrapping tokenizer for config: {name} -----")

        sp_model_path = os.path.join(tokenizer_dir, name, f'tokenizer_{name}.model')

        if not os.path.exists(sp_model_path):
            print(f"  - WARNING: SentencePiece model not found at '{sp_model_path}'. Skipping.")
            continue

        print(f"  - Loading SentencePiece model: {sp_model_path}")

        try:
            # 1. Load the raw sentencepiece model into a LlamaTokenizer object
            tokenizer = LlamaTokenizer(vocab_file=sp_model_path)
        except Exception as e:
            print(f"  - ERROR: Failed to load tokenizer for config {name}: {e}")
            continue

        # 2. Save the tokenizer using save_pretrained
        output_dir = os.path.join(tokenizer_dir, name)
        print(f"  - Saving in Transformers format to: {output_dir}")
        tokenizer.save_pretrained(output_dir)
        print(f"  - Successfully wrapped tokenizer for {name}.\n")

    print("\n----- All bilingual tokenizers have been wrapped. -----")


if __name__ == '__main__':
    main()