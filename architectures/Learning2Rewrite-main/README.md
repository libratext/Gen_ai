### L2R: Generalized LLM-Generated Text Detection

This repo contains a minimal implementation of the ACL 2025 paper Learning2Rewrite: Generalized LLM-Generated Text Detection ([link](https://arxiv.org/pdf/2408.04237?)).

### Run

1. Generate rewrites (saves `*_rewrite_{train,test}.json`):

```bash
python rewrite.py
```

2. Baseline RAIDAR features + MLP (expects rewrite outputs):

```bash
python raidar.py
```

3. Train L2R-style LoRA adapter (artifacts saved to `./models/llama-3-8b/`):

```bash
python train.py
```

### Checkpoint

The adaptor of the rewrite model is saved at [link](https://drive.google.com/file/d/1T_jK8T3qDwHLvDxcwVZosDvWL5Jjb9Wp/view?usp=drive_link). Download and unzip the file in the `./model/` directory.

### Citation

If you find this work helpful, please consider citing the paper.

```
@misc{li2025learningrewritegeneralizedllmgenerated,
      title={Learning to Rewrite: Generalized LLM-Generated Text Detection},
      author={Ran Li and Wei Hao and Weiliang Zhao and Junfeng Yang and Chengzhi Mao},
      year={2025},
      eprint={2408.04237},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2408.04237},
}
```
