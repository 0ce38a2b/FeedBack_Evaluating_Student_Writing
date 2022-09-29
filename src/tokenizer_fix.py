from pathlib import Path

transformers_path = Path("/users10/lyzhang/anaconda3/envs/apex/lib/python3.7/site-packages/transformers")

import shutil

input_dir = Path("/users10/lyzhang/model/deberta_fix")

convert_file = input_dir / "convert_slow_tokenizer.py"
conversion_path = transformers_path/convert_file.name

if conversion_path.exists():
    conversion_path.unlink()

shutil.copy(convert_file, transformers_path)
deberta_v2_path = transformers_path / "models" / "deberta_v2"

for filename in ['tokenization_deberta_v2.py', 'tokenization_deberta_v2_fast.py']:
    filepath = deberta_v2_path/filename
    if filepath.exists():
        filepath.unlink()

    shutil.copy(input_dir/filename, filepath)


from transformers.models.deberta_v2.tokenization_deberta_v2_fast import DebertaV2TokenizerFast


def get_deberta_tokenizer(name):
    return DebertaV2TokenizerFast.from_pretrained(name)


# tokenizer = DebertaV2TokenizerFast.from_pretrained("/users10/lyzhang/model/deberta_v3_large")
# text = "Hello World!"
# encoded_text = tokenizer.encode_plus(
#     text,
#     add_special_tokens=False,
#     return_offsets_mapping=True,
# )
# print(encoded_text)
