# Gen_ai

The goal of this repository is to compile every architectures, that we know of, that can detect a generated text by classifying it to either generated or human-written. A litterature work has been made over this subject to target specific architectures : [Raidar](https://github.com/cvlab-columbia/RaidarLLMDetect), [Mosaic](https://github.com/BaggerOfWords/MOSAIC), [Binoculars](https://github.com/ahans30/Binoculars), [DetectGPT](https://github.com/eric-mitchell/detect-gpt), [Fast-DetectGPT](https://github.com/baoguangsheng/fast-detect-gpt). There were some changes to those architectures to be able to test them on our own datasets.

## Usage
Make sure you install all the requirements by typing:
```
pip install -r requirements.txt
```
Make sure you add your API keys into the ```./config.yaml``` file.

### Raidar Usage
First Raidar rewrites the human or generated text using this command:

```
python ./architectures/RaidarLLMDetect-main/Arxiv/gen_arxiv_rewrite.py
```
This should take a few minutes to a few hours. You should get two new json file in ```./results/Raidar``` folder. 

And then, you apply the detection :
```
python ./architectures/RaidarLLMDetect-main/Arxiv/detect_arxiv_inv.py
```

### Binoculars Usage
To use Binoculars architecture, use this command :
```
python ./architectures/Binoculars-main/main_testonmydata.py
```
You should get a json file with all the predictions in ```./results/Binoculars``` 
### DetectGPT Usage
To use DetectGPT architecture, use this command :
```
python ./architectures/DetectGPT-main/infer_withmydata.py
```
You should get a json file with all the predictions in ```./results/DetectGPT``` 

### fast-DetectGPT Usage
To use fast-DetectGPT architecture, use this command :
```
python ./architectures/fast-detect-gpt-main/scripts/local_infer_withmydata.py --human_file_path ./datasets/human-micpro_original-fake_papers_train_part_public_extended.json --generated_file_path ./datasets/gen-micro_retracted-fake_papers_train_part_public_extended.json --output_file_path ./results/fast-DetectGPT/gpt-j-6B_kaggle_fast-detectgpt.json
```
You should get a json file with all the predictions in ```./results/fast-DetectGPT```

### Mosaic Usage

To use Mosaic architecture with the gpt2 models, use this command :
```
python ./architectures/MOSAIC-main/example_withmydata.py --model_set gpt2
```
You should get a json file with all the predictions in ```./results/Mosaic```

You can use other models like "llama": ["meta-llama/Llama-2-7b-chat-hf", "meta-llama/Llama-2-7b-hf"] with this command:
```
python ./architectures/MOSAIC-main/example_withmydata.py --model_set llama
```
Make sure you have access to those models with you Hugging face account and enter your token key in ```./config.yaml```

You can also use bigger models "tower": ["Unbabel/TowerBase-13B-v0.1", "TowerBase-7B-v0.1"] using this command:
```
python ./architectures/MOSAIC-main/example_withmydata.py --model_set tower
```

## Use the different architectures on your own dataset

