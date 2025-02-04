# R1-V: Reinforcing Super Generalization Ability in Vision Language Models with Less Than $3



1. We firstly reveal that **Reinforcement Learning with Verifiable Rewards (RLVR)** outperforms chain-of-thought supervised fine-tuning (CoT-SFT) in both **effectiveness and out-of-distribution (OOD) robustness** for vision language models.

2. In our experiment, we **incentivize** VLMs to learn **generalizable** visual counting abilities, rather than overfitting to the training set.

3. The 2B model outperforms the 72B model in OOD tests within just **100** training steps.

4. The training was conducted on 8 A100 GPUs for **30 minutes, costing $2.62**.

5. Codes, models, datasets, more details and **all open-source** resources will be shared (within CNY holidays).

**Contributors:** [Liang Chen](https://github.com/chenllliang) · [Lei Li](https://lilei-nlp.github.io) · [Haozhe Zhao](https://haozhezhao.github.io/) · [Yifan Song](https://github.com/Yifan-Song793)

---

[🤗 Train Dataset](https://huggingface.co/datasets/leonardPKU/clevr_cogen_a_train)

[🤗 R1-Distilled Visual Reasoning Dataset](https://huggingface.co/datasets/MMInstruction/Clevr_CoGenT_TrainA_R1)

Updates:

- 2025-02-03: We upload the training codebase.
- 2025-02-03: We curate and upload some verified Deepseek-R1 visual reasoning traces with some special tricks. Current training code does not rely on it, feel free to explore.


---





![image](./images/ood.png)

![image](./images/super_ood.png)

![image](./images/training.png)


## Requirements

```bash
cd src/open-r1-multimodal 
pip3 install -e ".[dev]"
pip3 install wandb==0.18.3
```

## Training

```bash
bash src/scripts/run_grpo_combine.sh
```

## Acknowledgements

We sincerely thank [DeepSeek](https://github.com/deepseek-ai/DeepSeek-R1), [Open-R1](https://github.com/huggingface/open-r1), [QwenVL](https://github.com/QwenLM/Qwen2.5-VL), [Open-R1-Multimodal](https://github.com/EvolvingLMMs-Lab/open-r1-multimodal), [CLEVR](https://cs.stanford.edu/people/jcjohns/clevr/), [SuperCLEVR](https://github.com/Lizw14/Super-CLEVR) for providing open source resources for us to build the project.


## Citation

```bib
@misc{chen2025r1v,
  author       = {Chen, Liang and Li, Lei and Zhao, Haozhe and Song, Yifan},
  title        = {R1-V: Reinforcing Super Generalization Ability in Vision-Language Models with Less Than \$3},
  howpublished = {\url{https://github.com/Deep-Agent/R1-V}},
  note         = {Accessed: 2025-02-02},
  year         = {2025}
}
```




