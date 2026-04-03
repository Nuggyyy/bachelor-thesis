from datasets import load_dataset
import re
import json

ds = load_dataset("google/fleurs", "is_is", trust_remote_code=True)

chars_to_remove_regex = r"[\[\],.?!\-;:\"“%‘”�'»«0-9/²–(\u200b)]"

def remove_special_characters(batch):
    batch["transcription"] = re.sub(chars_to_remove_regex, "", batch["transcription"]).lower()
    return batch

ds = ds.map(remove_special_characters)

# Build vocab
all_text = ""
for split in ds:
    all_text += " ".join(ds[split]["transcription"])

vocab_list = sorted(set(all_text))
vocab_dict = {v: k for k, v in enumerate(vocab_list)}

vocab_dict["[PAD]"] = vocab_dict[" "]
del vocab_dict[" "]
vocab_dict["|"] = len(vocab_dict)
vocab_dict["[UNK]"] = len(vocab_dict)

with open('vocab.json', 'w') as vocab_file:
    json.dump(vocab_dict, vocab_file)
