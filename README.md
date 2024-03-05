
<h1 align="center">
<img src="https://github.com/CriticBench/criticbench.github.io/raw/main/docs/static/images/criticbench_logo.png" width="80" alt="CriticBench" />
<br>
CriticBench: Benchmarking LLMs for Critique-Correct Reasoning
</h1>

<div align="center">

![](https://img.shields.io/badge/Code%20License-MIT-green)

</div>

<p align="center">
  <a href="https://criticbench.github.io/"><b>[🌐 Website]</b></a> •
  <a href="https://arxiv.org/abs/2402.14809"><b>[📜 Paper]</b></a> •
  <a href="https://huggingface.co/datasets/llm-agents/CriticBench"><b>[🤗 Dataset]</b></a> •
  <a href="https://github.com/CriticBench/CriticBench"><b>[🐱 GitHub]</b></a>
  <br>
  <!-- <a href="https://twitter.com/TODO"><b>[🐦 Twitter]</b></a> • -->
  <!-- <a href="#-quick-start">Quick Start</a> • -->
  <!-- <a href="#%EF%B8%8F-citation">Citation</a> -->
</p>

<p align="center">
Repo for "<a href="https://arxiv.org/abs/2402.14809" target="_blank">CriticBench: Benchmarking LLMs for Critique-Correct Reasoning</a>"
</p>

## 💡 Introduction
The ability of Large Language Models (LLMs) to critique and refine their reasoning is crucial for their application in evaluation, feedback provision, and self-improvement. This paper introduces **CriticBench, a comprehensive benchmark designed to assess LLMs' abilities to critique and rectify their reasoning across a variety of tasks.** CriticBench encompasses five reasoning domains: **mathematical, commonsense, symbolic, coding, and algorithmic.** It compiles 15 datasets and incorporates responses from three LLM families. Utilizing CriticBench, we evaluate and dissect the performance of 17 LLMs in generation, critique, and correction reasoning, i.e., GQC reasoning, and analyze the key factors affecting LLM critical reasoning.

Our findings reveal: (1) a linear relationship in GQC capabilities, with critique-focused training markedly enhancing performance; (2) a task-dependent variation in critique and correction effectiveness, with logic-oriented tasks being more amenable to correction; (3) GQC knowledge inconsistencies that decrease as model size increases; and (4) an intriguing inter-model critiquing pattern, where stronger models are better at critiquing weaker ones, while weaker models can surprisingly surpass stronger ones in their self-critique.
We hope these insights into the nuanced critique-correct reasoning of LLMs will foster further research in LLM critique and self-improvement.
<p align="center">
    <img src="https://github.com/CriticBench/criticbench.github.io/raw/main/docs/static/images/overview.png" width="1000">
        <br>
    <em>Figure 1: An overview for the CriticBench construction.</em>
</p>

## 🚀 Quick Start
### ⚙️ Setup
**Cloning the repository**
```sh
git clone git@github.com:CriticBench/CriticBench.git 
cd CriticBench/src
```
**Preparing conda env**
```sh
conda create -n critcbench python=3.10
conda activate criticbench
```
Install [torch](https://pytorch.org/get-started/locally/) that is compatible with your device, then install the required dependencies as follows:
```sh
pip install -r requirements.txt
```
### ⚖️ Evaluation
You can evaluation model's generation(G), critique(Q), correction(C) by the following command. 
#### Evaluate with specified model
Some models require access permissions, which can be set with the following commands:
```sh
export HUGGING_FACE_HUB_TOKEN=<Your Huggingface token>
export OPENAI_API_KEY=<Your OpenAI API key>
```
##### Huggingface model
```sh
python evaluate.py  \
    --available_gpus <GPU_IDs> \
    --tasks GQC \
    --prompt_type fs\
    --hf_model <model-name> \
    --enable_code_execution
```
#### Huggingface critique model
We provide support for [Auto-J](https://github.com/GAIR-NLP/auto-j) and [UltraCM](https://github.com/OpenBMB/UltraFeedback). You can evaluate these models with the following command.
```sh
python evaluate.py  \
    --available_gpus <GPU_IDs> \
    --tasks Q \
    --hf_critic_model <model-name> \
    --enable_code_execution
```
**OpenAI model**
```sh
python evaluate.py  \
    --tasks GQC \
    --prompt_type fs\
    --openai_model <model-name> \
    --enable_code_execution
```
+ `--tasks` specifies which task to evaluate, with the available options being:
  + `GQC` for a combination of generation, critique, and correction;
  + `QC` for critique and correction; 
  + `G`, `Q`, or `C` for generation, critique, or correction individually;
  + Note that correction tasks ("C") should be executed after critique tasks ("Q") or require a specified critique result file.
+ `--prompt_type` allows you to further specify the prompts for critique and correction used during evaluation:
  + `fs`: few-shot prompt for both critique and correction;
  + `zs-crit-cot`: zero-shot chain-of-thought prompt for critique;
  + `zs-crit-ao-1`, `zs-crit-ao-2` and `zs-crit-ao-3` represent three distinct types of zero-shot answer-only prompts for critique;
  + In correction, zero-shot prompts are all set to chain of thought (cot).
+ `--enable_code_execution` argument enables execution of code for generation and correction tasks
+ `--available_gpus` argument specifies which GPUs to use, identified by their IDs (e.g., `0,1`). 


#### Evaluate with existed result
You can specify paths to the existed result file using the `--existed_gen_file`, `--existed_crit_file` and `--existed_corr_file`. For accurate answer extraction, 
ensure `--prompt_type` aligns with your results. Here is an example:
```sh
python evaluate.py \
    --tasks GQC \
    --prompt_type fs \
    --enable_code_execution \
    --existed_gen_file <path to generation result file> \
    --existed_crit_file <path to critique result file> \
    --existed_corr_file <path to correction result file>
```
Here's an example of what a JSON line in a **generation result file** might look like:
```json lines
{
  "id": 0, 
  "final_prompt": "The final prompt for LLMs",
  "generation_result": "LLM's result for the generation task"
}
```

## ☕️ Citation
If you find this repository helpful, please consider citing our paper:

```
@misc{lin2024criticbench,
  title={CriticBench: Benchmarking LLMs for Critique-Correct Reasoning}, 
  author={Zicheng Lin and Zhibin Gou and Tian Liang and Ruilin Luo and Haowei Liu and Yujiu Yang},
  year={2024},
  eprint={2402.14809},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}
```