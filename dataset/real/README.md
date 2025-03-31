# Make your real-world Dataset

## Installation Instructions

1. Create conda environment:
```bash
conda env create -f environment.yml
conda activate real
```
rename folder -groundingdino to groundingdino and put it into the site-packages folder of your virtual environment

2. Set up directories and download model weights:
```bash
mkdir input_image
mkdir output
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
wget https://drive.google.com/file/d/1qobFYrI4eyIANfBSmYcGuWRaSIXfMOQ8/view?usp=sharing
wget https://huggingface.co/google-bert/bert-base-uncased/resolve/main/pytorch_model.bin?download=true
mv pytorch_model.bin bert-base-uncased
```

3. Set up IP-Adapter and DPT:
```bash
cd zest_code
git clone https://huggingface.co/h94/IP-Adapter
mv IP-Adapter/models models
mv IP-Adapter/sdxl_models sdxl_models
wget https://github.com/intel-isl/DPT/releases/download/1_0/dpt_hybrid-midas-501f0c75.pt
mv dpt_hybrid-midas-501f0c75.pt zest_code/DPT/weights
cd ..
```

## Execution Steps

### Step 1: Initial Processing
1. Prepare target material image:
   - Rename your target material image as `target.jpg`, `target.jpeg` or `target.png`
   - Place your target material image in the `input_image` folder

2. Run initial processing commands:
```bash
python grounded_sam.py --text_prompt=YOUR_PROMPT
python zest_code/DPT/run.py
```

### Step 2: Generate Intermediate Results
- Place your input images in `input_image` folder
```bash
python grounded_sam.py --text_prompt=YOUR_PROMPT
```
**Output explanation**: 
- The program will create a batch folder in the `output` directory (format: run+number, where number represents the batch number)
- Each input image will generate 3 intermediate result images, saved in the corresponding batch folder

### Step 3: Generate Final Results
1. Determine the batch number to process (check the batch folder names in the `output` directory)

2. Run final processing command:
```bash
python zest_code/zest.py --run=YOUR-RUN
```
**Output explanation**: 
- The final result will be saved in the same batch folder with the 3 intermediate result images
- The final result file will be clearly labeled as `output.png`
