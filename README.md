# Landmine-risk-prediction

### Instructions:
* Step 0: Creat a new conda environment is strongly recommended. Make sure you have these packages:
    * pytorch (cpu-version is sufficient)
    * gpytorch
    * pdpbox
    * matplotlib==3.2.2 (compatible with pdplot)
    * geopandas
    * scikit-learn
    * lightgbm
* **Step 1**: In prepare_parameters.ipynb will generate a json file in the root folder that saves all of your parameters. This notebook includes default and debug paramters to run all experiments. To customize parameters for training, run the last block in this notebook to get the flags for train.py. 
* **Step 2**: Train model. Open your terminal, and run the command given by the last block of prepare_parameters.ipynb. Ex, 
```
python train.py --load_json '...your_root_folder.../params_exp05052022113240.json'
```
* Your results (the best model, probabilities, feature selection intermediate steps, log) will be automatically saved in the exp/timestamp folder. The json file will be automatically moved into the exp/timestamp folder.
* If you attempt to overwrite old experiment results, you will receive "Are you sure to overwrite ....", please remember to type **y/n** in the Terminal.
* **Step 3**: Evaluate model on unseen region. You can specify the flags to run evaluation.py. For example, if you want to evaluate SVM using the model saved in exp/timestamp, run 
```
python evaluation.py --root (your root folder) --curr_time timestamp --model_name SVM
```
* Your evaluation results will be saved in exp/timestamp/eval folder.
* **Step 4**: Follow instructions in visualize_demo.ipynb to plot all necessary figures. 
