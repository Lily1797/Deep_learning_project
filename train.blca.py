import os
import torch
#from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from dataset.datasets import load_data, preprocess_data, split_data
from dataset.pytorch_dataset import PytorchDataset
from TabMDA.constructor import construct_tabmda_model
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, balanced_accuracy_score
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import random
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from utils import GLOBAL_SEED, set_seed, get_available_device, to_numpy
import warnings
warnings.filterwarnings("ignore")

# Define continuous and binary columns in patient information data
patient_continuous_columns = ['days_to_birth', 'height', 'weight', 'tobacco_smoking_history', 'number_pack_years_smoked', 'age_at_initial_pathologic_diagnosis']  
patient_binary_columns = ['gender', 'history_of_neoadjuvant_treatment', 'primary_lymph_node_presentation_assessment', 'lymphovascular_invasion_present', 'neoplasm_histologic_grade', 'history_non_muscle_invasive_blca', 'WHITE', 'BLACK OR AFRICAN AMERICAN', 'ASIAN', 'Manual', 'Office', 'Health', 'Teacher', 'Military', 'Retired', 'Transport', 'Homemaker', 'Chimique']

# Classifier dictionary
CLASSIFIERS = {
    "random_forest": lambda: RandomForestClassifier(n_estimators=200, max_depth=3, min_samples_leaf=2, random_state=0),
    "logistic_regression": lambda: LogisticRegression(max_iter=1000, random_state=0),
    "xgboost": lambda: XGBClassifier(n_estimators=200, learning_rate=0.3, n_jobs=2, verbosity=0, max_depth=3, random_state=0, device='cuda' if torch.cuda.is_available() else 'cpu'),
    "neural_network": lambda: MLPClassifier(hidden_layer_sizes=(128,), max_iter=1000, random_state=42)
}

def load_and_preprocess_data(data_choice, omic_path, patient_info_path, labels_path, 
                             patient_continuous_cols=patient_continuous_columns, patient_binary_cols=patient_binary_columns):
    """
    Load data based on user choice (omic, clinical, or combined) and preprocess it.
    For omics data, all columns are assumed to be continuous and extracted dynamically.
    For patient data, column types (continuous/binary) must be specified.
    """
    # Load labels
    labels_df = pd.read_csv(labels_path, header=0, sep=';')
    labels_df['labels'] = labels_df.drop('ID', axis=1).idxmax(axis=1)  # Get the column name of the max value
    labels_df['labels'] = labels_df['labels'].apply(lambda x: ['LumP', 'Ba/Sq', 'LumU', 'Stroma-rich', 'LumNS', 'NE-like'].index(x))
    labels = labels_df[['ID', 'labels']]
    labels.set_index("ID", inplace=True)

    # Load data
    if data_choice == "omic":
        data = pd.read_csv(omic_path, header=0, sep=';')
        data.set_index("ID", inplace=True)
        omics_cols = data.columns  # Automatically use all columns as continuous
    elif data_choice == "clinical":
        data = pd.read_csv(patient_info_path, header=0, sep=';')
        data.set_index("ID", inplace=True)
    elif data_choice == "combined":
        omic_data = pd.read_csv(omic_path, header=0, sep=';')
        omic_data.set_index("ID", inplace=True)
        omics_cols = omic_data.columns  # Automatically use all columns as continuous
        patient_data = pd.read_csv(patient_info_path, header=0, sep=';')
        patient_data.set_index("ID", inplace=True)
        data = pd.concat([omic_data, patient_data], axis=1)
    else:
        raise ValueError("Invalid data choice. Choose 'omic', 'clinical', or 'combined'.")

    # Align data with labels
    data = data.join(labels, how="inner")

    # Preprocess data
    X = data.drop(columns=labels.columns, errors="ignore")
    y = data[labels.columns]

    X_processed = pd.DataFrame(index=X.index)

    # Process omics columns (if present)
    if data_choice in ["omic", "combined"]:
        omics_imputer = SimpleImputer(strategy="mean")
        X_omics = pd.DataFrame(omics_imputer.fit_transform(X[omics_cols]), 
                               columns=omics_cols, index=X.index)
        scaler = StandardScaler()
        X_omics[omics_cols] = scaler.fit_transform(X_omics[omics_cols])
        X_processed = pd.concat([X_processed, X_omics], axis=1)

    # Process continuous patient columns
    if data_choice in ["clinical", "combined"] and patient_continuous_cols:
        valid_patient_continuous_cols = [col for col in patient_continuous_cols if col in X.columns]
        patient_cont_imputer = SimpleImputer(strategy="mean")
        X_patient_cont = pd.DataFrame(patient_cont_imputer.fit_transform(X[valid_patient_continuous_cols]), 
                                      columns=valid_patient_continuous_cols, index=X.index)
        scaler = StandardScaler()
        X_patient_cont[valid_patient_continuous_cols] = scaler.fit_transform(X_patient_cont[valid_patient_continuous_cols])
        X_processed = pd.concat([X_processed, X_patient_cont], axis=1)

    # Process binary patient columns
    if data_choice in ["clinical", "combined"] and patient_binary_cols:
        valid_patient_binary_cols = [col for col in patient_binary_cols if col in X.columns]
        patient_bin_imputer = SimpleImputer(strategy="most_frequent")
        X_patient_bin = pd.DataFrame(patient_bin_imputer.fit_transform(X[valid_patient_binary_cols]), 
                                     columns=valid_patient_binary_cols, index=X.index)
        X_processed = pd.concat([X_processed, X_patient_bin], axis=1)

    # Return preprocessed features and labels
    return X_processed, y.values

def split_data(X, y, test_size=0.2, val_size=0.16, random_state=42):
    """
    Split data into training, validation, and test sets.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size_adjusted, random_state=random_state)
    return X_train, X_val, X_test, y_train, y_val, y_test

def pca_data_structure(train, val, test, step, output_dir=None):
    # Perform PCA on the data
    pca = PCA(n_components=2)  # Reduce to 2D for visualization
    X_all = np.concatenate([train, val, test], axis=0)
    X_pca = pca.fit_transform(X_all)

    # Create a DataFrame for seaborn to handle the plot
    pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
    
    # Add labels to the DataFrame for each subset (train, val, test)
    pca_df['Set'] = ['Train'] * len(train) + ['Validation'] * len(val) + ['Test'] * len(test)
    
    # Create the seaborn plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="Set", style="Set", palette="Set1", alpha=0.6)

    plt.title(f"PCA of {step} Data")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    
    # Save the plot
    pca_plot_path = os.path.join(output_dir, f"pca_{step}.png")
    plt.savefig(pca_plot_path)
    plt.close()

def pca_one_set(X, y, set_name, output_dir=None):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
    df_pca["Label"] = y

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df_pca, x="PC1", y="PC2", hue="Label", palette="viridis", alpha=0.7)
    plt.title(f"PCA of {set_name} Data")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(title="Label", bbox_to_anchor=(1.05, 1), loc='upper left')

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        pca_plot_path = os.path.join(output_dir, f"pca_{set_name.lower()}.png")
        plt.savefig(pca_plot_path)
    else:
        plt.show()
    plt.close()

def main(args):
    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory if it doesn't exist
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # ===================================================
    # Load and preprocess data 
    # ===================================================
    X, y = load_and_preprocess_data(args.data_choice, args.omic_path, args.patient_info_path, args.labels_path)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    
    pca_data_structure(X_train, X_val, X_test, "Pre-TabMDA", args.output_dir)

    X_train, X_val, X_test = to_numpy(X_train), to_numpy(X_val), to_numpy(X_test)

    # ===================================================
    # Apply the model
    # ===================================================
    TabMDA_model = construct_tabmda_model(classifier=None,freeze_encoder=True,device=device)

    # Make data as Tensor and put on the device
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)
    
    print(f"[Data] Original train shape: {X_train.shape}")
    print(f"[Data] Original val shape: {X_val.shape}")
    print(f"[Data] Original test shape: {X_test.shape}")

    # Encode the data
    smote_params = {"k": args.smote_k, "rounds": args.smote_rounds} if args.smote_TabMDA else None
    X_train_enc, y_train_enc = TabMDA_model.encode_batch(
        batch={"x": X_train_tensor, "y": y_train_tensor, "x_context": [X_train_tensor], "y_context": [y_train_tensor]},
        context_subsetting_params={"num_contexts": args.num_contexts, "context_size": args.context_size},
        smote_params=smote_params
    )
    X_val_enc, y_val_enc = TabMDA_model.encode_batch(
        batch={"x": X_val_tensor, "y": y_val_tensor,"x_context": [X_train_tensor], "y_context": [y_train_tensor]},
        context_subsetting_params={"num_contexts": args.num_contexts_val, "context_size": 1 if args.num_contexts_val == 1 else args.context_size},
        smote_params=None
    )
    X_test_enc, y_test_enc = TabMDA_model.encode_batch(
        batch={"x": X_test_tensor, "y": y_test_tensor, "x_context": [X_train_tensor], "y_context": [y_train_tensor]},
        context_subsetting_params={"num_contexts": args.num_contexts_test, "context_size": 1 if args.num_contexts_test == 1 else args.context_size},
        smote_params=None
    )

    print(f"[Data] Encoded train shape: {X_train_enc.shape}")
    print(f"[Data] Encoded test shape: {X_test_enc.shape}")
    print(f"[Data] Encoded val shape: {X_val_enc.shape}")

    # ===================================================
    # Evaluate the model with the classifier
    # ===================================================
    classifier = CLASSIFIERS[args.classifier_model]()
    classifier.fit(X_train_enc, y_train_enc)

    y_train_pred = to_numpy(classifier.predict(X_train_enc))
    y_val_pred = to_numpy(classifier.predict(X_val_enc))
    y_test_pred = to_numpy(classifier.predict(X_test_enc))
    y_test_pred_proba = to_numpy(classifier.predict_proba(X_test_enc))

    # Save classification reports
    classification_report_train = classification_report(y_train_enc, y_train_pred)
    classification_report_val = classification_report(y_val_enc, y_val_pred)
    classification_report_test = classification_report(y_test_enc, y_test_pred)

    with open(os.path.join(output_dir, "classification_report_train.txt"), "w") as f:
        f.write(classification_report_train)

    with open(os.path.join(output_dir, "classification_report_val.txt"), "w") as f:
        f.write(classification_report_val)
    
    with open(os.path.join(output_dir, "classification_report_test.txt"), "w") as f:
        f.write(classification_report_test)

    print("Train Classification Report:\n", classification_report_train)
    print("Validation Classification Report:\n", classification_report_val)
    print("Test Classification Report:\n", classification_report_test)

    # Save confusion matrix plot
    cm = confusion_matrix(y_test_enc, y_test_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.arange(args.num_classes), yticklabels=np.arange(args.num_classes))
    plt.title("Confusion Matrix - Test Set")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    
    cm_plot_path = os.path.join(output_dir, "confusion_matrix_test.png")
    plt.savefig(cm_plot_path)
    plt.close()

    # Visualize PCA results
    pca_data_structure(X_train_enc, X_val_enc, X_test_enc, "Post-TabMDA", args.output_dir)
    pca_one_set(X_train_enc, y_train_pred, "Train", args.output_dir)
    pca_one_set(X_val_enc, y_val_pred, "Validation", args.output_dir)
    pca_one_set(X_test_enc, y_test_pred, "Test", args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TabMDA Training and Evaluation")
    parser.add_argument("--data_choice", type=str, default="omic", help="Dataset to use (e.g., tcga, custom)")
    parser.add_argument("--omic_path", type=str, required=False, help="Path to protein expression file")
    parser.add_argument("--patient_info_path", type=str, required=False, help="Path to patient information file")
    parser.add_argument("--labels_path", type=str, required=True, help="Path to labels file")
    parser.add_argument("--classifier_model", type=str, choices=CLASSIFIERS.keys(), required=True, help="Classifier model to use")
    parser.add_argument("--num_contexts", type=int, default=10, help="Number of contexts for data augmentation")
    parser.add_argument("--context_size", type=float, default=0.5, help="Context size")
    parser.add_argument("--num_contexts_val", type=int, default=10, help="Number of val contexts")
    parser.add_argument("--num_contexts_test", type=int, default=10, help="Number of test contexts")
    parser.add_argument("--num_classes", type=int, default=4, help="Number of classes for classification")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for regularization")
    parser.add_argument("--train_epochs", type=int, default=10, help="Number of epochs for training")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output results")
    parser.add_argument('--smote_TabMDA',action='store_true',dest='smote_TabMDA',help='True if you want to apply SMOTE augmentation on TabMDA.')
    parser.set_defaults(smote_TabMDA=False)
    parser.add_argument('--smote_rounds',default=1,type=int,help='Rounds of SMOTE')
    parser.add_argument('--smote_k',default=3,type=int,help='Number of K-nearest neighbors for SMOTE')
    args = parser.parse_args()
    main(args)
