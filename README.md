# Mc-ADFAL 
A multimodal model for chemical-receptor affinity prediction
requirements：
python 3.7.13
rdkit 2019.03.1.0
xgboost 
pytorch
sklearn
dgl

Raw data can also be accessed from zenodo (https://zenodo.org/records/15034076).

Data Preparation
Once the environment is set up, you need to prepare the whole data files (.csv) and place them in the ..data\davis\raw folder. The data file should follow these format requirements:

Data Columns:

compound_iso_smiles: SMILES string for the compound.

target_sequence: Protein sequence.

affinity: Binding affinity.

If you need to change the column names, please modify them in the preprocessing.py file.

Model Training
Once the data files are ready, you can start training the model. In the command line, navigate to the Model directory and run the following commands:

cd C:\Model

python preprocessing.py

python train.py --dataset dataset --save_model

Replace --dataset with the name of your dataset.

The --save_model flag ensures that the trained model will be saved after training.


Model Testing
After training the model, you can test it using the following command. Make sure you have the trained model file ready, and specify the model path:

python test.py --dataset dataset --model_path model_path

--dataset: Specify the name of the dataset you are using.

--model_path: Path to the saved model.



