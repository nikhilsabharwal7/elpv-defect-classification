# Solar Cell Defect Classification Using ELPV

This project classifies photovoltaic cells as defective or non-defective from electroluminescence images. It uses pretrained ConvNeXt-Tiny and EfficientNetV2-S models, ordinal supervision, contrast enhancement, and texture features.

**Recorded test accuracy — 91.92%**, with 603 of 656 test images classified correctly.

[Open the Jupyter notebook](ELPV_Fine_Tuned.ipynb)

## Author

Nikhil Sabharwal  
Roll number — 2026PAD2531  
M.Tech in Artificial Intelligence and Data Science (AIDS)  
Under the guidance of Dr. Ankush Jain, Department of CSE at NSUT

## Public dataset

The project uses the [public ELPV dataset](https://github.com/zae-bayern/elpv-dataset).

The dataset contains 2,624 grayscale images of solar cells, each measuring 300 × 300 pixels, extracted from 44 photovoltaic modules. Each image has a defect-confidence annotation and a monocrystalline or polycrystalline technology label.

| Confidence label | Binary target used in this project |
|---|---|
| 0 | Non-defective |
| 1/3 | Non-defective |
| 2/3 | Defective |
| 1 | Defective |

Download the ZIP using **Code → Download ZIP** in the dataset repository. Save it locally as `elpv-dataset-master.zip` beside the notebook.

The ELPV images are distributed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International license. Refer to the [dataset repository's license information](https://github.com/zae-bayern/elpv-dataset#license).

## What the notebook does

1. Loads and validates the ELPV images and annotations.
2. Creates a development set of 1,968 images and a test set of 656 images.
3. Fine-tunes ConvNeXt-Tiny and EfficientNetV2-S using four development folds.
4. Learns the binary decision together with ordered confidence targets.
5. Combines neural predictions, image texture measurements, and cell technology using a logistic classifier.
6. Refits the selected backbones on the full development set.
7. Reports test results and displays training curves and confusion matrices.

## Run locally

Use a Python environment with Jupyter and the following packages:

```bash
pip install torch torchvision numpy pandas pillow scipy scikit-learn matplotlib jupyterlab
jupyter lab
```

Open `ELPV_Fine_Tuned.ipynb` and run the cells in order. Set `DATASET` in the setup cell if your dataset is stored elsewhere. A CUDA-enabled GPU makes model training faster. Official pretrained weights download on first use.

The embedded outputs record the original experiments. Running the training cells produces a new run whose results can vary with the environment.

## Results and evaluation

| Configuration | Correct predictions | Test accuracy |
|---|---:|---:|
| Fine-tuned fold ensemble | 602 / 656 | 91.77% |
| Full-development refits | 601 / 656 | 91.62% |
| Blend of fold models and full-development refits | **603 / 656** | **91.92%** |

The best observed result contained 5 false alarms and 48 missed defective cells. The same cell-level test split was examined during development, so the result is exploratory. The public release omits source-module IDs, and the split can contain cells from the same module in both development and test.

## References

- [ELPV dataset repository](https://github.com/zae-bayern/elpv-dataset)
- Ebied et al. (2025), *Advanced deep learning modeling to enhance detection of defective photovoltaic cells in electroluminescence images*. [Scientific Reports](https://doi.org/10.1038/s41598-025-14478-y)
- Deitsch et al. (2019), *Automatic classification of defective photovoltaic module cells in electroluminescence images*. [Solar Energy](https://doi.org/10.1016/j.solener.2019.02.067)
- Buerhop-Lutz et al. (2018), *A benchmark for visual identification of defective solar cells in electroluminescence imagery*. [EU PVSEC](https://doi.org/10.4229/35thEUPVSEC20182018-5CV.3.15)
