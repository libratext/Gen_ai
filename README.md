# Gen_ai

The goal of this repository is to compile every architectures, that we know of, that can detect a generated text by classifying it to either generated or human-written. A litterature work has been made over this subject to target specific architectures : [Raidar](https://github.com/cvlab-columbia/RaidarLLMDetect), [Mosaic](https://github.com/BaggerOfWords/MOSAIC), [Binoculars](https://github.com/ahans30/Binoculars), [DetectGPT](https://github.com/eric-mitchell/detect-gpt), [Fast-DetectGPT](https://github.com/baoguangsheng/fast-detect-gpt)). There were some changes to those architectures to be able to test them on our own datasets.

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

To use Mosaic architecture, use this command :
```
python ./architectures/MOSAIC-main/example_withmydata.py
```
You should get a json file with all the predictions in ```./results/Mosaic```

# Results

| Architectures | Precision | Recall | F1 | Results file |
| --- | ----------- | ----------- | ----------- | ----------- |
| Binoculars | 0.5 | 1.0 | 0.66 | [Results](./results/Binoculars/falcon-rw-1b_Binoculars_gen_human-micro_retracted-fake_papers_train_part_public_extended.json) |
| DetectGPT | 0 | 0 | 0 | [Results](./results/DetectGPT/kaggle_evaluation_metrics.json) |
| fast-DetectGPT | Text | Text | Title | Title |
| Mosaic | Text | Text | Title | Title |
| Raidar | Text | Text | Title | Title |