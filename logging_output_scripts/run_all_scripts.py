from logging_output_scripts.violin_plots import create_violin_plots
from logging_output_scripts.calc_bay import calc_bayes
from logging_output_scripts.summary_csv import create_summary_csv
from logging_output_scripts.latex_tabulars import write_complexity, write_mse, single_table

if __name__ == '__main__':
    create_violin_plots()
    calc_bayes()
    create_summary_csv()
    write_complexity()
    write_mse()
    single_table()
