# Pancreatic Cancer Survival Prediction Using Digital Pathology and MIL

Jeong Hoon Lee (Stanford University School of Medicine) and Jongmin Sung (Roche)

## Overview

This project employs a deep learning pipeline for predicting patient survival in pancreatic cancer using whole slide images (WSIs). The approach builds on our previous work by adapting and extending our multiple instance learning (MIL) framework. We use the TCGA PDAC dataset for training and external data from the National Cancer Institute for validation. In addition, external datasets from different tissue types (CHOL, COAD, LUAD, ESCA, LIHC, BRCA) are incorporated to capture more generalized pattern information.

## Datasets

- **Training Dataset:**  
  - **TCGA PDAC:** WSIs from the TCGA pancreatic cancer dataset are used for model training.

- **Primary External Validation:**  
  - **National Cancer Institute PDAC Data:** An independent set of WSIs is used for external validation of survival predictions.

- **Additional External Data:**  
  - WSIs from other tissues including Cholangiocarcinoma (CHOL), Colon Adenocarcinoma (COAD), Lung Adenocarcinoma (LUAD), Esophageal Carcinoma (ESCA), Liver Hepatocellular Carcinoma (LIHC), and Breast Cancer (BRCA) are used to enhance generalization.

## Methodology

1. **Patch Extraction & Feature Extraction:**  
   - WSIs are divided into smaller patches (e.g., 224×224 pixels).  
   - The digital pathology foundation model (**Uni**), fine-tuned from DINOv2, extracts low-dimensional feature vectors from each patch.

2. **Aggregation via Multiple Instance Learning (MIL):**  
   - The extracted patch features are aggregated using an attention-based MIL framework into a slide-level representation.  
   - The attention mechanism learns to weight patches based on their importance for predicting survival.

3. **Slide-Level Prediction:**  
   - The aggregated slide-level features are used to predict patient survival.  
   - Slide-level annotations (cancer/survival outcomes) supervise training, and the model learns which patches are most predictive.

4. **Interpretability:**  
   - The attention scores are used to generate heatmaps that highlight the most informative patches.  
   - These highlighted regions can be reviewed by doctors to verify whether the features correspond to known cancer-related textures or cell patterns.

## MIL Methods Explored

We evaluated several MIL architectures, including:
- Multiple Instance Learning (MIL)
- ACMIL
- CLAM_MB
- CLAM_SB
- TransMIL
- DSMIL
- ABMIL
- GABMIL
- MeanMIL
- MaxMIL

## Loss Functions

To find the optimal model, we explored various loss functions:
- Cox Proportional Hazards Loss (Coxph)
- Rank Loss
- Mean Squared Error (MSE) Loss
- SurvMLE Loss

## Models Tested

The following digital pathology models were compared:
- **UNI (Faisal Mahmood)** – *Best performing model*
- **GigaPath** (Microsoft, Washington)
- **Lunit-DINO**

## Results & Conclusion

- **Performance:**  
  The UNI model outperformed the other tested models in predicting pancreatic cancer patient survival from WSIs.
  
- **Generalization:**  
  Incorporating external datasets from diverse tissue types improved the model's ability to learn general patterns, enhancing prognostic accuracy.
  
- **Interpretability:**  
  The attention mechanism provides heatmaps highlighting key patches that drive the survival prediction, bridging automated analysis and pathological assessment.

## Future Work

- **Refinement of MIL Architectures:**  
  Further optimization of patch-level feature selection and aggregation.
  
- **Multi-Modal Integration:**  
  Integrate histopathology features with clinical and molecular data to enhance prognostic modeling.
  
- **Expanded Validation:**  
  Validate the model on larger, multi-center datasets to ensure robustness and clinical utility.

## References

1. **Cox, D. R.** (1972). Regression Models and Life-Tables. *Journal of the Royal Statistical Society: Series B (Methodological)*. [Link](https://www.jstor.org/stable/2985181)
2. **Caron, M., et al.** (2021). Emerging Properties in Self-Supervised Vision Transformers. *arXiv preprint arXiv:2104.14294*. [Link](https://arxiv.org/abs/2104.14294)
3. **Ilse, M., Tomczak, J. M., & Welling, M.** (2018). Attention-based Deep Multiple Instance Learning. In *Proceedings of the 35th International Conference on Machine Learning (ICML)*. [Link](https://proceedings.mlr.press/v80/ilse18a.html)
4. **Campanella, G., et al.** (2019). Clinical-grade computational pathology using weakly supervised deep learning on whole slide images. *Nature Medicine*. [Link](https://www.nature.com/articles/s41591-019-0508-1)
5. **Mahmood, F., et al.** (2023). UNI: A Digital Pathology Foundation Model. *BioRxiv*. [Link](https://www.biorxiv.org/)
