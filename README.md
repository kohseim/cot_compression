<h1 align="center">
  Zipping the Thought: When and How Compressed Reasoning Data Works in LLM Post-Training
</h1>

<p align="center">
  Kohsei Matsutani
  ·
  Gouki Minegishi
  ·
  Takeshi Kojima
  ·
  Yusuke Iwasawa
  ·
  Yutaka Matsuo
  <br/>
  The University of Tokyo
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2605.28008">
    <img src="https://img.shields.io/badge/arXiv-2605.28008-b31b1b.svg" alt="arXiv">
  </a>
</p>

## Setup

```bash
conda create -n zipping python=3.11 -y
conda activate zipping

pip install -r requirements.txt
pip install -e ./LLaMA-Factory
pip install -e ./verl
```

## Experiments

### Synthetic Tasks

```bash
# Explicit CoT
python src/datagenerationworker.py \
  --numprocs 16 --total 40000 --mod 23 --number_range 5 \
  --listoperations 10 11 12 \
  --output_dir data/raw --flat_output \
  --modes forward


# Composed CoT ＆ Implicit CoT
python src/datagenerationworker.py \
  --numprocs 16 --total 40000 --mod 23 --number_range 5 \
  --listoperations 10 11 12 \
  --output_dir data/raw --flat_output \
  --all_nests 2 4 --variant_types composed implicit
```

### SFT ＆ RLVR

**SFT**

```bash
DATA_ROOT=data/raw MODE=explicit bash scripts/run_sft.sh
```

**RLVR**

```bash
python src/prepare_rlvr_data.py \
  --input-dir data/raw --op-min 10 --op-max 12 \
  --output-dir data/rlvr

MODEL_PATH=saves/arithmetic_sft/Qwen-Qwen2.5-3B/explicit \
TRAIN_FILE=data/rlvr/train.parquet \
VAL_FILE=data/rlvr/val.parquet \
bash scripts/run_rlvr.sh
```

### Evaluation

Greedy pass@1 on a held-out JSONL.

```bash
CKPT=saves/arithmetic_sft/Qwen-Qwen2.5-3B/explicit \
DATA=data/eval/op10.jsonl \
bash scripts/run_eval.sh
```



## Acknowledgements

This codebase builds on the [LlamaFactory](https://github.com/hiyouga/LlamaFactory), [verl](https://github.com/verl-project/verl), and  [Interplay-LM-Reasoning](https://github.com/Interplay-LM-Reasoning/Interplay-LM-Reasoning) repositories.


## Citation

```bibtex
@article{matsutani2026zipping,
  title={Zipping the Thought: When and How Compressed Reasoning Data Works in {LLM} Post-Training},
  author={Kohsei Matsutani, Gouki Minegishi, Takeshi Kojima, Yusuke Iwasawa, Yutaka Matsuo},
  journal={arXiv preprint arXiv:2605.28008},
  year={2026}
}
```
