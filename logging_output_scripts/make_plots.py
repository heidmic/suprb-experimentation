import json
import overtime
import violin_plots
import summary_csv
import latex_tabulars

if __name__ == "__main__":
    isClassifier = True
    if isClassifier:
        with open('logging_output_scripts/config_classification.json') as f:
            config = json.load(f)
    else:
        with open('logging_output_scripts/config_regression.json') as f:
            config = json.load(f)
    overtime.create_plots(metric_name="elitist_complexity", isClassifier=isClassifier)
    violin_plots.create_violin_plots(metric_name="elitist_complexity", isClassifier=isClassifier)
    
    if isClassifier:
        violin_plots.create_violin_plots(metric_name="training_score", isClassifier=True)
        overtime.create_plots(metric_name="training_score", isClassifier=True)
    else:
        violin_plots.create_violin_plots(metric_name="test_r2", isClassifier=False)
        overtime.create_plots(metric_name="training_score", isClassifier=False)
    summary_csv.create_summary_csv(isClassifier=isClassifier)
    
    for model in config['model_names']:
        summary_csv.create_summary_csv(base_model=model, isClassifier=isClassifier)
    latex_tabulars.create_latex_tables(isClassifier=isClassifier)
