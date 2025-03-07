import json
import overtime
import violin_plots
import summary_csv
import latex_tabulars

if __name__ == "__main__":
    isClass = True
    if isClass:
        with open('logging_output_scripts/config_class.json') as f:
            config = json.load(f)
    else:
        with open('logging_output_scripts/config.json') as f:
            config = json.load(f)
    overtime.create_plots(metric_name="elitist_complexity", isClass=isClass)
    violin_plots.create_violin_plots(metric_name="elitist_complexity", isClass=isClass)
    
    if isClass:
        violin_plots.create_violin_plots(metric_name="training_score", isClass=True)
        overtime.create_plots(metric_name="training_score", isClass=True)
    else:
        violin_plots.create_violin_plots(metric_name="test_neg_mean_squared_error", isClass=False)
        overtime.create_plots(metric_name="training_score", isClass=False)
    summary_csv.create_summary_csv(isClass=isClass)
    
    for model in config['model_names']:
        summary_csv.create_summary_csv(base_model=model, isClass=isClass)
    latex_tabulars.create_latex_tables(isClass=isClass)
