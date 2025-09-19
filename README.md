# Gen_ai

The goal of this repository is to compile every architectures, that we know of, that can detect a generated text by classifying it to either generated or human-written. A litterature work has been made over this subject to target specific architectures : [Raidar](https://github.com/cvlab-columbia/RaidarLLMDetect), [Mosaic](https://github.com/BaggerOfWords/MOSAIC), [Binoculars](https://github.com/ahans30/Binoculars), [DetectGPT](https://github.com/eric-mitchell/detect-gpt), [Fast-DetectGPT](https://github.com/baoguangsheng/fast-detect-gpt). There were some changes to those architectures to be able to test them on our own datasets.

## Usage
Make sure you install all the requirements by typing :
```
pip install -r requirements.txt
```
Make sure you add your API keys into the ```./config.yaml``` file.

### Raidar Usage
First Raidar rewrites the human or generated text using this command :

```
python ./architectures/RaidarLLMDetect-main/Arxiv/gen_arxiv_rewrite.py
```
This should take a few minutes to a few hours. You should get two new json file in ```./results/Raidar``` folder. 

And then, you apply the detection :
```
python ./architectures/RaidarLLMDetect-main/Arxiv/detect_arxiv_inv.py
```
You can also use Raidar only on a specific text. Make sure the text is not too small or too big. Make sure you add ```./architectures/RaidarLLMDetect-main/Arxiv/ada_rewrite_arxiv_GPT_inv.json``` and ```./architectures/RaidarLLMDetect-main/Arxiv/ada_rewrite_arxiv_human_inv.json``` to the ```config.yaml``` file for Raidar model training.

```
python ./architectures/RaidarLLMDetect-main/Arxiv/predict_raidar.py --input_text "This is a test."
```

### Binoculars Usage
To use Binoculars architecture with a smaller model (tiiuae/falcon-rw-1b & tiiuae/falcon-rw-1b ), use this command :
```
python ./architectures/Binoculars-main/main_testonmydata.py --model small
```
To use a bigger model (tiiuae/falcon-7b & tiiuae/falcon-7b-instruct), use this command :
```
python ./architectures/Binoculars-main/main_testonmydata.py --model big
```
You should get a json file with all the predictions in ```./results/Binoculars```

You can also use Binoculars only on a specific text. Make sure the text is not too small or too big. You can use a smaller model (tiiuae/falcon-rw-1b & tiiuae/falcon-rw-1b ) or a bigger model (tiiuae/falcon-7b & tiiuae/falcon-7b-instruct)
```
python ./architectures/Binoculars-main/predict_binoculars.py \
--input_text "This is a test." \
--model small
```

### DetectGPT Usage
To use DetectGPT architecture, use this command :
```
python ./architectures/DetectGPT-main/infer_withmydata.py
```
You should get a json file with all the predictions in ```./results/DetectGPT``` 

You can also use DetectGPT only on a specific text. Make sure the text is not too small or too big.
```
python ./architectures/DetectGPT-main/predict_detectGPT.py \
--input_text "This is a test."
```

### fast-DetectGPT Usage
To use fast-DetectGPT architecture, use this command :
```
python ./architectures/fast-detect-gpt-main/scripts/local_infer_withmydata.py \
--human_file_path ./datasets/human-micpro_original-fake_papers_train_part_public_extended.json \
--generated_file_path ./datasets/gen-micro_retracted-fake_papers_train_part_public_extended.json \
--output_file_path ./results/fast-DetectGPT/gpt-j-6B_kaggle_fast-detectgpt.json
```
You should get a json file with all the predictions in ```./results/fast-DetectGPT```. By default, it uses a smaller model (gpt-j-6B and gpt-neo-2.7B).

You can use this command to use a bigger model (falcon-7b & falcon-7b-instruct) :
```
python ./architectures/fast-detect-gpt-main/scripts/local_infer_withmydata.py \
--sampling_model_name falcon-7b \
--scoring_model_name falcon-7b-instruct \
--human_file_path ./datasets/human-micpro_original-fake_papers_train_part_public_extended.json \
--generated_file_path ./datasets/gen-micro_retracted-fake_papers_train_part_public_extended.json \
--output_file_path ./results/fast-DetectGPT/falcon-7b_kaggle_fast-detectgpt.json
```
You can also use fast-DetectGPT only on a specific text. Make sure the text is not too small or too big. You can use specific sampling ("gpt-j-6B"/"gpt-neo-2.7B"/"falcon-7b") and scoring models ("gpt-neo-2.7B"/"falcon-7b-instruct").
```
python ./architectures/fast-detect-gpt-main/scripts/predict_fast-detectgpt.py \
--input_text "This is a test."
```

### Mosaic Usage

To use Mosaic architecture with the gpt2 models, use this command :
```
python ./architectures/MOSAIC-main/example_withmydata.py --model_set gpt2
```
You should get a json file with all the predictions in ```./results/Mosaic```

You can use other models like "llama": ["meta-llama/Llama-2-7b-chat-hf", "meta-llama/Llama-2-7b-hf"] with this command :
```
python ./architectures/MOSAIC-main/example_withmydata.py --model_set llama
```
Make sure you have access to those models with you Hugging face account and enter your token key in ```./config.yaml```

You can also use bigger models "tower": ["Unbabel/TowerBase-13B-v0.1", "TowerBase-7B-v0.1"] using this command :
```
python ./architectures/MOSAIC-main/example_withmydata.py --model_set tower
```
You can also use Mosaic only on a specific text. Make sure the text is not too small or too big. You can use specific models using this list:
```
"gpt2": ["openai-community/gpt2-medium", "openai-community/gpt2"],
"llama": ["meta-llama/Llama-2-7b-chat-hf", "meta-llama/Llama-2-7b-hf"],  # Ensure Hugging Face token is set inside the config.yaml file
"tower": ["Unbabel/TowerBase-13B-v0.1", "Unbabel/TowerBase-7B-v0.1"]
```
```
python ./architectures/MOSAIC-main/predict_mosaic.py \
--input_text "This is a test." \
--model gpt2
```

## Use the different architectures on your own dataset

As soon as you write your own dataset paths into the ```./config.yaml``` file, all the pipelines will priorise your dataset apart from fast-DetectGPT which has a parameter usage where you can point to your own dataset paths in the command line when calling the script. Make sure your own dataset has the exact same format as the default datasets from this repository in ```./datasets```.
