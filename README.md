# Where am I? Cross-View Geo-localization with Natural Language Descriptions

<!--
**CVG-Text/CVG-Text** is a ✨ _special_ ✨ repository because its `README.md` (this file) appears on your GitHub profile.

Here are some ideas to get you started:

- 🔭 I’m currently working on ...
- 🌱 I’m currently learning ...
- 👯 I’m looking to collaborate on ...
- 🤔 I’m looking for help with ...
- 💬 Ask me about ...
- 📫 How to reach me: ...
- 😄 Pronouns: ...
- ⚡ Fun fact: ...
-->

## Environment Setup

```bash
conda create -n CVG-Text python=3.9 -y
conda activate CVG-Text
pip install -r requirements.txt
```

## Dataset Download and Path Configuration

The images and annotation files for CVG-Text can be found at [https://huggingface.co/CVG-Text/CVG-Text](https://huggingface.co/datasets/CVG-Text/CVG-Text)

After downloading, update the `/path/to/dataset/` in `./config.yaml` with the actual dataset paths.

## Testing
To retrieve satellite images (sat) using NewYork-mixed (panoramic + single-view) text and the Ours model, run:
```bash
python zeroshot.py --version NewYork-mixed --img_type sat --model CLIP-L/14@336 --expand
```
You can also evaluate specific checkpoint by setting `--checkpoint {your_checkpoint_path}`
For more examples, please refer to the script in `./scripts/evaluate.sh`.

## Training
To train the Ours model on Brisbane-mixed and OSM datasets, use the following command:
```bash
python -m torch.distributed.run --nproc_per_node=4 finetune.py --lr 1e-5 --batch_size 128 --epochs 40 --version Brisbane-mixed --model CLIP-L/14@336 --expand --img_type sat --logging
```
The `--logging` flag determines whether to save log files and model checkpoints.

## Model Checkpoints

Our model checkpoints are available at: [https://huggingface.co/CVG-Text/CrossText2Loc](https://huggingface.co/CVG-Text/CrossText2Loc)
