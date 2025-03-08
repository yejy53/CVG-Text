# Where am I? Cross-View Geo-localization with Natural Language Descriptions

[Junyan Ye](https://yejy53.github.io/), [Honglin Lin](https://lhl3341.github.io/), Leyan Ou, Dairong Chen, Zihao Wang,  [Conghui He](https://conghui.github.io/), [Weijia Li](https://liweijia.github.io/)

Sun Yat-Sen University, Shanghai AI Laboratory, Sensetime Research, Wuhan University


[![arXiv](https://img.shields.io/badge/Arxiv-2410.02761-b31b1b.svg?logo=arXiv)]([https://arxiv.org/abs/2410.02761](https://arxiv.org/abs/2412.17007)) 
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow)](https://github.com/zhipeixu/FakeShield/blob/main/LICENSE) 
[![Home Page](https://img.shields.io/badge/Project_Page-CVG-Text.svg)](https://yejy53.github.io/CVG-Text/)


<!--
**CVG-Text/CVG-Text** is a ✨ _special_ ✨ repository because its `README.md` (this file) appears on your GitHub profile.
[![hf_space](https://img.shields.io/badge/🤗-Huggingface%20Checkpoint-blue.svg)](https://huggingface.co/datasets/CVG-Text/CVG-Text)
Here are some ideas to get you started:

- 🔭 I’m currently working on ...
- 🌱 I’m currently learning ...
- 👯 I’m looking to collaborate on ...
- 🤔 I’m looking for help with ...
- 💬 Ask me about ...
- 📫 How to reach me: ...
- 😄 Pronouns: ...
- ⚡ Fun fact: ...
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fzhipeixu%2FFakeShield&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)
[https://huggingface.co/CVG-Text/CVG-Text](https://huggingface.co/datasets/CVG-Text/CVG-Text)
[https://huggingface.co/CVG-Text/CrossText2Loc](https://huggingface.co/CVG-Text/CrossText2Loc)

-->



## 📰 News

* **[2024.12]**  🔥 We have released **Where am I? Cross-View Geo-localization with Natural Language Descriptions**. Check out the [paper](https://arxiv.org/abs/2412.17007). The code and dataset are coming soon




## 🏆 Contributions

**Novel mission setting：** We introduce and formalize the Cross-View Geo-localization task based on natural language descriptions, utilizing scene text descriptions to retrieve corresponding OSM or satellite images for geographical localization.

**Dataset Contribution：** We propose CVG-Text, a dataset with well-aligned street-views, satellite images, OSM, and text descriptions across three cities and over 30,000 coordinates. Additionally a progressive scene text generation framework based on LMM is presented, which reduces vague descriptions and generates high-quality scene text.

**New retrieve methods：** We introduce CrossText2Loc, a novel text localization method that excels in handling long texts and interpretability. It achieves over a 10\% improvement in Top-1 recall compared to existing methods, while offering retrieval reasoning beyond similarity scores.


## 🛠️ Requirements and Installation

Ensure your environment meets the following requirements:

```bash
conda create -n CVG-Text python=3.9 -y
conda activate CVG-Text
pip install -r requirements.txt
```

## 🤗 Dataset Download and Path Configuration

**Dataset：** The images and annotation files for CVG-Text can be found at 

**Path Configuration：** After downloading, update the `/path/to/dataset/` in `./config.yaml` with the actual dataset paths.

**Model Checkpoints：** Our model checkpoints are available at: 
## 🚀 Quick Start
To retrieve satellite images (sat) using NewYork-mixed (panoramic + single-view) text and the Ours model, run:
```bash
python zeroshot.py --version NewYork-mixed --img_type sat --model CLIP-L/14@336 --expand
```
You can also evaluate specific checkpoint by setting `--checkpoint {your_checkpoint_path}`
For more examples, please refer to the script in `./scripts/evaluate.sh`.

## 🏋️‍♂️ Train
To train the Ours model on Brisbane-mixed and OSM datasets, use the following command:
```bash
python -m torch.distributed.run --nproc_per_node=4 finetune.py --lr 1e-5 --batch_size 128 --epochs 40 --version Brisbane-mixed --model CLIP-L/14@336 --expand --img_type sat --logging
```
The `--logging` flag determines whether to save log files and model checkpoints.

## BibTeX 🙏

If you have any questions, be free to contact with me! 
```
@article{ye2024cross,
  title={Where am I? Cross-View Geo-localization with Natural Language Descriptions},
  author={Ye, Junyan and Lin, Honglin and Ou, Leyan and Chen, Dairong and Wang, Zihao and He, Conghui and Li, Weijia},
  journal={arXiv preprint arXiv:2412.17007},
  year={2024}
}
```


