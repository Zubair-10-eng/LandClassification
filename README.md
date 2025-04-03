# Land Classification with YOLOv8, ResNet-50, and ViT on EuroSAT Dataset 🌍✨

![Image](https://raw.githubusercontent.com/Slygriyrsk/LandClassification/master/Result/EuroSAT.png)

Hey there! 👋 Welcome to our awesome project on classifying land types from satellite images using the EuroSAT dataset! We’re a group of students from the Indian Institute of Information Technology Dharwad, and we’ve been messing around with three cool models—YOLOv8, ResNet-50, and Vision Transformer (ViT)—to figure out stuff like Forests 🌲, Highways 🛣️, or Crops 🌾 from space pics. We did it all on free Google Colab, which was a wild ride! 🎢 Our goal? Smash at least 95% accuracy and learn heaps about deep learning and satellites. 🚀

## What’s This Project About? 🗺️
We grabbed 27,000 tiny satellite images (64x64 pixels) from EuroSAT, split into 10 land-use classes like AnnualCrop, Forest, and SeaLake. We trained three models to sort them out:
- **YOLOv8 Small**: Fast as lightning ⚡ but hit a snag.
- **ResNet-50**: Deep and dependable, like a trusty sidekick. 🦸
- **ViT**: New and brainy with its attention magic. 🧠

We battled Colab’s GPU timeouts ⏰, wrestled with bugs 🐞, and came out with some sweet results! Plus, we whipped up visuals like GradCAM and a flowchart to show off our work. This README has it all—results, files, how to run it, why it’s cool, and what’s next! 😎

## Why This Project Rocks 🌟
- **Real-World Vibes**: Land classification helps with farming 🌾, city planning 🏙️, and watching nature—like spotting shrinking forests.
- **Budget Win**: We rocked free Colab and still got killer results—no fancy gear needed! 💸
- **Model Mashup**: Mixing YOLOv8, ResNet-50, and ViT on EuroSAT is pretty fresh—we’re trailblazers! 🔥
- **Learning FTW**: Bugs and crashes taught us coding and problem-solving ninja skills. 🥷
- **Eye Candy**: Our pics and charts (check ‘em below!) make it fun to see what’s up—like where ResNet-50’s looking! 👀

## Our Results 🎯
Here’s the scoop on how our models did—full stats ahead!

### Overall Performance 📊
Trained on 21,600 images, tested on 1340 validation pics (out of 5,400 total—testing all was sloooow 🐢). Check it out:

| Model       | Accuracy (Train) | Accuracy (Test) | Precision | Recall | F1-Score | Notes                              |
|-------------|------------------|-----------------|-----------|--------|----------|------------------------------------|
| YOLOv8 Small| 95.9% (500 imgs) | 11.07% (1340)   | 1.31%     | 11.07% | 2.35%    | Bug crashed testing—used train stats 🐞 |
| ResNet-50   | Not recorded     | 97.2% (1340)    | 97.41%    | 97.2%  | 97.26%   | Steady and strong 💪               |
| ViT         | Not recorded     | 100% (1340)     | 100%      | 100%   | 100%     | Perfect score—wow! 🌟             |

- **YOLOv8 Small**:  
  - Trained 11 epochs (wanted 15, but Colab said “nah” ⏳). Hit 95.9% on a mini training set (500 images)—looked good!  
  - Testing tanked at 11.07% due to a `TypeError` bug—precision (1.31%) and F1 (2.35%) were sad too. Used training stats instead. 😕

- **ResNet-50**:  
  - Trained 15 epochs, tested on 1340 images, and scored a solid 97.2%.  
  - Precision (97.41%), recall (97.2%), and F1 (97.26%) were tight—just a few slip-ups! 😊

- **ViT**:  
  - Trained 10 epochs and nailed 100% on 1340 test images—insane!  
  - Precision, recall, and F1 all 100%. Maybe too good for a small set, but that attention trick is clutch! 🎉

### Class-Specific Results (AnnualCrop) 🌾
We dug into AnnualCrop to see how each model handled it:

| Model       | Precision | Recall | F1-Score | AUC   |
|-------------|-----------|--------|----------|-------|
| YOLOv8      | 0%        | 0%     | 0%       | 0.45  |
| ResNet-50   | 98.5%     | 97.8%  | 98.1%    | 0.99  |
| ViT         | 100%      | 100%   | 100%     | 1.00  |

- **YOLOv8**: Zilch—0% across the board in testing. AUC (0.45) says it was guessing. That bug strikes again! 😭
- **ResNet-50**: Killed it with 98.5% precision, 97.8% recall, 98.1% F1. AUC of 0.99—super confident! 👍
- **ViT**: Perfect 100%—precision, recall, F1, and AUC (1.00). It’s like it’s cheating! 😂

### What We Think 🤔
- YOLOv8 had potential but that bug wrecked it—gotta fix it!
- ResNet-50 was our rock, smashing our 95% goal.
- ViT’s 100% is wild—wonder if it’ll hold up on more pics?

**Visuals Alert!** Check these out:
- ![Image](https://raw.githubusercontent.com/Zubair-10-eng/LandClassification/master/Result/confusion_matrices.png) - Where models got confused!
- ![Image](https://raw.githubusercontent.com/Zubair-10-eng/LandClassification/master/Result/sample_predictions_collage.png) - What each model guessed!
- ![Image](https://raw.githubusercontent.com/Zubair-10-eng/LandClassification/master/Result/resnet_gradcam.png) - Where ResNet-50 looked!
- ![Image](https://raw.githubusercontent.com/Zubair-10-eng/LandClassification/master/Result/flowchart.png) - Our whole process!

## Files in This Repo 📂
Here’s what we’ve packed in here:

1. **`land_classification.ipynb`**  
   - Our Colab notebook with all the code—data prep, training, testing, and visuals. 📜

2. **`eurosat.yaml`**  
   - YOLOv8’s config file—points to train (21,600 imgs) and val (5,400 imgs) folders + 10 classes. ⚙️

3. **`requirements.txt`**  
   - All the Python goodies we used (`torch`, `ultralytics`, etc.). Install with `pip install -r requirements.txt`. 🛠️

4. **`final_model.pt`**  
   - YOLOv8’s trained weights—grabbed from Colab’s `best.pt`. 💾

5. **`resnet_model.pth`**  
   - ResNet-50’s weights after 15 epochs. 💿

6. **`vit_model/`**  
   - Folder with ViT’s pretrained weights—saved with `save_pretrained()`. 📁

7. **`images/` Folder**  
   - Our visual goodies:  
     - `confusion_matrices.png`  
     - `sample_predictions_collage.png`  
     - `gradcam_resnet.png`  
     - `project_flowchart.png` 🎨

8. **`report.tex`**  
   - Our LaTeX report—abstract, methods, results, all that jazz. Compile it with `pdflatex`! 📝

9. **`README.md`**  
   - This file—your guide to our project! 📖

**Heads Up**: EuroSAT dataset’s too big (27,000 imgs)! Grab it from [phelber/EuroSAT](https://github.com/phelber/EuroSAT) and stick it in Drive.

## How to Run This Project 🚀
Ready to dive in? Here’s how:

1. **Get the Dataset**  
   - Snag EuroSAT from [GitHub](https://github.com/phelber/EuroSAT).  
   - Unzip and upload to Drive (e.g., `/content/drive/MyDrive/EuroSAT/2750`). 📥

2. **Set Up Colab**  
   - Hit up [Google Colab](https://colab.research.google.com).  
   - Upload `land_classification.ipynb`.  
   - Grab a T4 GPU: `Runtime > Change runtime type > T4 GPU`. ⚡

3. **Install Stuff**  
   - Run this in Colab:
     ```bash
     !pip install -r requirements.txt
     ```
   - Or manually: `torch`, `ultralytics`, `transformers`, `numpy`, `matplotlib`. 🛠️

4. **Prep the Data**  
   - Tweak `eurosat.yaml` with your Drive paths (e.g., `/content/drive/MyDrive/EuroSAT/train`).  
   - Run the notebook’s prep code—splits data with symlinks. 📊

5. **Train and Test**  
   - Run the notebook cells to train (hours—watch timeouts ⏰!).  
   - Use our models (`final_model.pt`, `resnet_model.pth`, `vit_model/`) to skip training. ✅

6. **See the Results**  
   - Notebook spits out stats and saves pics to `images/`.  
   - Peek at our pre-made visuals too! 👀

7. **Compile the Report**  
   - Grab `report.tex`, open in Overleaf or a LaTeX editor, and run `pdflatex`. 📜

## Why This Project is the Best 🌈
- **Skills Galore**: Coding, debugging, data wrangling—we leveled up! 🧠  
- **Top Scores**: 97.2% and 100% with free tools? Yes, please! 🏆  
- **Fresh Take**: Three models on EuroSAT with Colab—kinda unique! ✨  
- **Team Spirit**: Saharsh, Zubair, Pranshul, and Rakesh rocked it together! 🤝  
- **Show-Off Ready**: Visuals and report = instant brag material. 📸

## Future Plans 🔮
We’ve got big dreams:
- **Fix YOLOv8**: Squash that bug—maybe a path or format fix? 🐞  
- **More Training**: Colab Pro for longer runs—past 11 epochs! ⏳  
- **Full Test**: Try all 5,400 val images for real scores. 📈  
- **Model Mash**: Blend ResNet-50 and ViT—super-model time! ⚡  
- **Live Action**: YOLOv8 for real-time satellite stuff? Let’s see! 🌍

---
**Links:**  
- Dataset: [phelber/EuroSAT](https://github.com/phelber/EuroSAT)  
- Colab: [Our Notebook](https://colab.research.google.com/drive/[your-link-here])