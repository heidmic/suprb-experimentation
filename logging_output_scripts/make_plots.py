import json
import overtime
import violin_plots
import summary_csv

if __name__ == "__main__":
    isClass = False
    if isClass:
        with open('logging_output_scripts/config_class.json') as f:
            config = json.load(f)
    else:
        with open('logging_output_scripts/config.json') as f:
            config = json.load(f)
    overtime.create_plots(metric_name="elitist_complexity", isClass=isClass)
    violin_plots.create_violin_plots(metric_name="elitist_complexity", isClass=isClass)
    if isClass:
        violin_plots.create_violin_plots(metric_name="elitist_error", isClass=True)
        overtime.create_plots(metric_name="elitist_error", isClass=True)
    else:
        violin_plots.create_violin_plots(metric_name="test_neg_mean_squared_error", isClass=False)
        overtime.create_plots(metric_name="elitist_error", isClass=False)
    summary_csv.create_summary_csv(isClass=isClass)
    for model in config['model_names']:
        summary_csv.create_summary_csv(base_model=model, isClass=isClass)

