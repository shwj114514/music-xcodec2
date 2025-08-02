# music-xcodec2

This repository refactors the original [X-Codec 2.0](https://github.com/zhenye234/X-Codec-2.0) implementation, inspired by the code structure of [stable-audio-tools](https://github.com/Stability-AI/stable-audio-tools). I provide training scripts with semantic supervision from [MuQ](https://github.com/tencent-ailab/MuQ), and [BEATs](https://github.com/microsoft/unilm/tree/master/beats).


## 🔧 Installation
My Python version is `3.10.14`, Install required packages via:
`pip install -r requirements.txt`
or manually:
```bash
pip install torch==2.4.0
pip torchao==0.9.0
pip torchtune==0.3.1
# plus any other dependencies listed in requirements.txt
```

## 📂 Prepare Data
First, download your audio data into a folder, e.g., test_audio.

Then, generate the JSONL filelist by running:
```bash
python scripts/get_filelist.py
```
This will create a JSONL file such as `config/data/test_audio.jsonl`.

Next, create the dataset configuration JSON file at `config/dataset/test_audio.json`. You can include multiple JSONL datasets, for example:

```json
{
    "dataset_type": "audio_dir",
    "datasets": [
        {
            "id": "test1",
            "path": "config/data/test_audio.jsonl"
        },
          {
            "id": "test2",
            "path": "config/data/test_audio.jsonl"
        }
    ],
    "random_crop": true
}
```
## 🚀 Training

Model configurations are specified in:

- `config/trainer/muq_25hz.json`
- `config/trainer/beats_50hz.json`

> Note: The `codebooksize` is set as `65536` in `xcodec2/vq/codec_decoder_vocos.py`.

To start training:
```bash
bash train_muq_25hz.sh
```
> Important: The MUQ model involves mel spectrogram extraction. Ensure that SSL extraction runs under FP32 precision.
## 🎵 Inference
```bash
bash infer.sh
```


## ✅ TODO
- [ ] Release training checkpoint
- [ ] Support inference with original XCodec2(w2v-bert-2.0) checkpoints
- [ ] Autoregressive (AR) code

## 🙏 Acknowledgements

- [X-Codec-2.0](https://github.com/zhenye234/X-Codec-2.0), the original implementation from which much of the code was adapted.
- [stable-audio-tools](https://github.com/Stability-AI/stable-audio-tools) for the Lightning-style project structure
- Folder [beats/](./beats/)  are directly adapted from the [BEATs](https://github.com/microsoft/unilm/tree/master/beats)  repository, which is licensed under the [MIT License](https://github.com/microsoft/unilm/blob/master/LICENSE)