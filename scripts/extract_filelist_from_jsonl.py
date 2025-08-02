import json

input_jsonl = "configs/data/fma_full.jsonl"
output_txt = 'fma_full.txt'

with open(input_jsonl, 'r') as jsonl_file:
    with open(output_txt, 'w') as txt_file:
        for line in jsonl_file:
            data = json.loads(line.strip())
            path = data.get('path', 'No path')
            prompt = data.get('prompt', 'No prompt')
            txt_file.write(f'{path}\t{prompt}\n')