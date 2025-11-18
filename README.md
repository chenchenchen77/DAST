# DAST

[DAST](https://aclanthology.org/2025.findings-acl.1055) is a simple yet effective method that leverages the LLM’s intrinsic understanding of contextual relevance to guide compression.

## Environment
```bash
conda create dast python=3.10.14

conda activate dast

# You may need to adjust the cuda version
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers deepspeed accelerate datasets peft pandas seaborn rouge fuzzywuzzy jieba python-Levenshtein
pip install flash-attn --no-build-isolation
```



## Evaluation Script
```
Download the trained model Llama-2-7b-chat from: https://cloud.tsinghua.edu.cn/d/ea67f973203046a7be31/

bash eval.sh
```

**NOTE**: It's okay to see warnings like `This is a friendly reminder - the current text generation call will exceed the model's predefined maximum length (32768). Depending on the model, you may observe exceptions, performance degradation, or nothing at all.` Just ignore it.


## Data
You should download the data for training & evaluation then untar the file at anywhere you prefer from dast, e.g. `/data`:
```bash
# feel free to alternate /data to your prefered location
wget https://huggingface.co/datasets/namespace-Pt/projects/resolve/main/long-llm.tar.gz?download=true -O /data/long-llm.tar.gz

cd /data
tar -xzvf long-llm.tar.gz
```


## Training
Our training methodology is based on Activation Beacon (UltraGist). Since our method focuses on query-related tasks, we processed the multi-turn dataset into a single-turn format.

Reference Repository (UltraGist): https://github.com/namespace-Pt/UltraGist

## Citation
if you find this repository useful for your research, please give us a star ⭐ and cite our work:

```
@inproceedings{chen-etal-2025-dast,
    title = "{DAST}: Context-Aware Compression in {LLM}s via Dynamic Allocation of Soft Tokens",
    author = "Chen, Shaoshen  and
      Li, Yangning  and
      Xu, Zishan  and
      Zeng, Yongqin  and
      Wu, Shunlong  and
      Hu, Xinshuo  and
      Shan, Zifei  and
      Su, Xin  and
      Tang, Jiwei  and
      Li, Yinghui  and
      Zheng, Hai-Tao",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2025",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-acl.1055/",
    doi = "10.18653/v1/2025.findings-acl.1055",
    pages = "20544--20552",
    ISBN = "979-8-89176-256-5",
    abstract = "Large Language Models (LLMs) face computational inefficiencies and redundant processing when handling long context inputs, prompting a focus on compression techniques. While existing semantic vector-based compression methods achieve promising performance, these methods fail to account for the intrinsic information density variations between context chunks, instead allocating soft tokens uniformly across context chunks. This uniform distribution inevitably diminishes allocation to information-critical regions. To address this, we propose Dynamic Allocation of Soft Tokens (DAST), a simple yet effective method that leverages the LLM{'}s intrinsic understanding of contextual relevance to guide compression. DAST combines perplexity-based local information with attention-driven global information to dynamically allocate soft tokens to the informative-rich chunks, enabling effective, context-aware compression. Experimental results across multiple benchmarks demonstrate that DAST surpasses state-of-the-art methods."
}

```

## Acknowledgement
We appreciate the following GitHub repos a lot for their valuable code and efforts.

Ultragist (https://github.com/namespace-Pt/UltraGist)

Activation beacon (https://github.com/FlagOpen/FlagEmbedding/tree/master/research/Long_LLM/activation_beacon, Long Context Compression with Activation Beacon, ICLR2025)
