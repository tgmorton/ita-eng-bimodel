# 03_tokenize_datasets.py
import os
from glob import glob

import sentencepiece as spm
from datasets import load_dataset


def tokenize_and_save(tokenizer, files_to_tokenize, output_dir, split_name):
    if not files_to_tokenize:
        print(f"  - WARNING: No data files found for '{split_name}' split. Skipping.")
        return

    print(f"  - Tokenizing {len(files_to_tokenize)} file(s) for '{split_name}' split...")
    print(f"  - Output directory: {output_dir}")

    dataset = load_dataset('text', data_files=files_to_tokenize, split='train')

    def tokenize_function(examples):
        encoded = [
            tokenizer.encode(text, out_type=int) for text in examples['text']
        ]
        return {'input_ids': encoded}

    tokenized_dataset = dataset.map(
        tokenize_function, batched=True, num_proc=os.cpu_count(), remove_columns=['text']
    )
    os.makedirs(output_dir, exist_ok=True)
    tokenized_dataset.save_to_disk(output_dir)
    print(f"  - Successfully saved '{split_name}' split.\n")


def main():
    print("\n--- Running Script 03: Tokenizing Datasets for Bilingual Models ---")
    base_dir = os.path.expanduser('~/ita-eng-bimodel')
    processed_dir = os.path.join(base_dir, 'data', 'processed')
    tokenizer_dir = os.path.join(base_dir, 'tokenizer')
    tokenized_dir = os.path.join(base_dir, 'data', 'tokenized')
    print(f"Reading processed from:  {processed_dir}")
    print(f"Reading tokenizers from: {tokenizer_dir}")
    print(f"Writing tokenized to:    {tokenized_dir}\n")

    bilingual_configs = [
        {'name': '10_25_it_eng', 'l1_lang': 'italian', 'l1_size': '10M', 'l2_lang': 'english', 'l2_size': '25M'},
        {'name': '25_25_it_eng', 'l1_lang': 'italian', 'l1_size': '25M', 'l2_lang': 'english', 'l2_size': '25M'},
        {'name': '50_25_it_eng', 'l1_lang': 'italian', 'l1_size': '50M', 'l2_lang': 'english', 'l2_size': '25M'},
        {'name': '10_25_eng_it', 'l1_lang': 'english', 'l1_size': '10M', 'l2_lang': 'italian', 'l2_size': '25M'},
        {'name': '25_25_eng_it', 'l1_lang': 'english', 'l1_size': '25M', 'l2_lang': 'italian', 'l2_size': '25M'},
        {'name': '50_25_eng_it', 'l1_lang': 'english', 'l1_size': '50M', 'l2_lang': 'italian', 'l2_size': '25M'},
    ]

    for config in bilingual_configs:
        name = config['name']
        print(f"----- Processing dataset for config: {name} -----")

        tokenizer_model_path = os.path.join(tokenizer_dir, name, f'tokenizer_{name}.model')
        if not os.path.exists(tokenizer_model_path):
            print(f"  - FATAL ERROR: Tokenizer not found at '{tokenizer_model_path}'. Skipping.")
            continue

        tokenizer = spm.SentencePieceProcessor()
        tokenizer.load(tokenizer_model_path)
        print(f"  - Loaded tokenizer: {os.path.basename(tokenizer_model_path)}")

        # --- Tokenize L1 and L2 training datasets ---
        l1_files = glob(os.path.join(processed_dir, config['l1_lang'], config['l1_size'], '*.train'))
        output_l1_dir = os.path.join(tokenized_dir, name, 'l1_train')
        tokenize_and_save(tokenizer, l1_files, output_l1_dir, f"{config['l1_lang']}-{config['l1_size']}")

        l2_files = glob(os.path.join(processed_dir, config['l2_lang'], config['l2_size'], '*.train'))
        output_l2_dir = os.path.join(tokenized_dir, name, 'l2_train')
        tokenize_and_save(tokenizer, l2_files, output_l2_dir, f"{config['l2_lang']}-{config['l2_size']}")

        # --- Tokenize Italian and English test datasets ---
        it_test_files = glob(os.path.join(processed_dir, 'italian', 'test_data_reduced', '*.test'))
        output_it_test_dir = os.path.join(tokenized_dir, name, 'italian_test')
        tokenize_and_save(tokenizer, it_test_files, output_it_test_dir, 'italian_test')

        en_test_files = glob(os.path.join(processed_dir, 'english', 'test_data', '*.test'))
        output_en_test_dir = os.path.join(tokenized_dir, name, 'english_test')
        tokenize_and_save(tokenizer, en_test_files, output_en_test_dir, 'english_test')

    print("\n----- All datasets have been tokenized and saved. -----")


if __name__ == '__main__':
    main()