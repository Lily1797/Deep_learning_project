# Deep learning project
Deep Learning and Artificial Intelligence course
___
# TabMDA Paper Review and Application to Genomic Data

In this project, we review and analyse the paper *"TabMDA: Data Augmentation for Tabular Domains"* by Andrei Margeloiu et al. (2024). Our goal is to apply the TabMDA approach to  multiomic and clinical data, exploring how well this manifold-based data augmentation strategy performs on different types of omics data compared to the original datasets used in the study.

Specifically, we aim to investigate the differences between the types of tabular data used in TabMDA and our diverse omic dataset, which includes:
- Protein expression data
- Gene expression data
- miRNA expression data
- Patient clinical information
- labelled data

We hope to understand the difference between the different data they used in their paper vs. the omic data that we use to extract meaningful insights into how well the augmentation framework generalizes to different omics data types and discuss any unique challenges presented by these biological datasets.

### TabMDA
TabMDA utilises the in-context learning (ICL) capabilities of any pre-trained tabular in-context model to increase the robustness and performance of any downstream machine learning models. In a nutshell, this approach augments the training dataset by introducing diversity through label-invariant transformations in the manifold space, leveraging multiple contexts for the embedder. 

### Project
The dataset used in this project is TCGA-BLCA dataset containing omics and clinical data retrieved from TCGA Project
TCGA-BLCA is a dataset that groups cases related to bladder cancer and its subtypes.
Disease types:
...

The goal of this project is to be able to correctly classify each subtype of bladder cancer using omics and clinical data.
We first augmented the input dataset through TabMDA and then train the downstream prediction model using our custom python script 'train.blca.py'. 
```
# Example usage
python train.blca.py \
	--data_choice clinical # Choices: omic, clinical, combined \
	--omic_path /path/to/input/omic/file \
	--patient_info_path /path/to/input/patient/infor/file \
	--labels_path /path/to/labels/file \ 
	--classifier_model xgboost # Classifiers: logistic_regression, random_forest, xgboost, neural_network \
	--train_epochs 10 \
	--context_size 0.5 \
	--num_contexts 20 \
	--num_classes 6 \
    --output_dir /path/to/output/dir
```
