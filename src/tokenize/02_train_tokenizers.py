# 02_train_tokenizers.py
import os
from glob import glob

import sentencepiece as spm


def main():
    print("\n--- Running Script 02: Training Bilingual Tokenizers ---")
    base_dir = os.path.expanduser('~/ita-eng-bimodel')
    processed_dir = os.path.join(base_dir, 'data', 'processed')
    tokenizer_dir = os.path.join(base_dir, 'tokenizer')
    print(f"Reading processed from:   {processed_dir}")
    print(f"Writing tokenizers to:    {tokenizer_dir}\n")

    bilingual_configs = [
        {'name': '10_25_it_eng', 'it_size': '10M', 'en_size': '25M'},
        {'name': '25_25_it_eng', 'it_size': '25M', 'en_size': '25M'},
        {'name': '50_25_it_eng', 'it_size': '50M', 'en_size': '25M'},
        {'name': '10_25_eng_it', 'it_size': '25M', 'en_size': '10M'},
        {'name': '25_25_eng_it', 'it_size': '25M', 'en_size': '25M'},
        {'name': '50_25_eng_it', 'it_size': '25M', 'en_size': '50M'},
    ]

    for config in bilingual_configs:
        name = config['name']
        it_size = config['it_size']
        en_size = config['en_size']

        print(f"----- Preparing Tokenizer for config: {name} -----")

        it_train_files = glob(os.path.join(processed_dir, 'italian', it_size, '*.train'))
        en_train_files = glob(os.path.join(processed_dir, 'english', en_size, '*.train'))
        it_test_files = glob(os.path.join(processed_dir, 'italian', 'test_data_reduced', '*.test'))
        en_test_files = glob(os.path.join(processed_dir, 'english', 'test_data', '*.test'))


        print(f"  - Found {len(it_train_files)} Italian train files for {it_size}.")
        print(f"  - Found {len(en_train_files)} English train files for {en_size}.")
        print(f"  - Found {len(it_test_files)} Italian test files.")
        print(f"  - Found {len(en_test_files)} English test files.")

        all_input_files = it_train_files + en_train_files + it_test_files + en_test_files

        if not all_input_files:
            print(f"  - WARNING: No files found for config {name}. Skipping.")
            continue

        output_model_dir = os.path.join(tokenizer_dir, name)
        os.makedirs(output_model_dir, exist_ok=True)
        model_prefix = os.path.join(output_model_dir, f'tokenizer_{name}')

        args = {
            'input': ",".join(all_input_files),
            'model_prefix': model_prefix,
            'vocab_size': 32000,
            'model_type': 'unigram',
            'max_sentence_length': 8192,
            'character_coverage': 1.0,
            'hard_vocab_limit': 'false',
        }
        arg_string = ' '.join([f'--{key}={value}' for key, value in args.items()])

        print(f"  - Starting training for tokenizer_{name}.model...")
        spm.SentencePieceTrainer.train(arg_string)
        print(f"  - Successfully trained tokenizer for {name}.\n")

    print("\n----- All bilingual tokenizer training complete. -----")


if __name__ == '__main__':
    main()