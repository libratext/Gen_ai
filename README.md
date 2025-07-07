# Gen_ai

The goal of this repository is to compile every architectures, that we know of, that can detect a generated text by classifying it to either generated or human-written. A litterature work has been made ofver this subject to target specific architecures : [Raidar](https://github.com/cvlab-columbia/RaidarLLMDetect), [Mosaic](https://github.com/BaggerOfWords/MOSAIC), [Binoculars](https://github.com/ahans30/Binoculars), [DetectGPT](https://github.com/eric-mitchell/detect-gpt), [Fast-DetectGPT](https://github.com/baoguangsheng/fast-detect-gpt)). There were some changes to those architectures to be able to test them on our own found datasets.

## Usage
Make sure you install all the requirements by typing:
```
pip install -r requirements.txt
```

Make sure you add your API keys into the```./config.yaml``` file.
