# Landmine-risk-prediction

Instruction of running models:
Step 1: Prepare_parameters.ipynb, get the path of json, which includes all of your parameters. Then run the last block in this file to get the "python train.py" configuration. (sample: python train.py --load_json '~/params_exp05052022113240.json')
Step 2: Open your terminal, and run (python train.py --load_json '~/params_exp05052022113240.json') jason, your result  will be automatically saved in the "exp/current_time" folder.
Step 3: "evaluation.py" You can specify the flags based on evaluation.py. For example, if you want to evaluate SVM using the model saved in exp/05052022113240, run "evaluation.py --root (your root folder) --curr_time 05052022113240 --model_name SVM"
Step 4: Follow instructions in visualize_demo.ipynb to plot all necessary figures.
