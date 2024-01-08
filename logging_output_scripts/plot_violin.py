import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from glob import glob

sns.set_style("whitegrid")
sns.set_theme(style="whitegrid",
              font="Times New Roman",
              font_scale=1,
              rc={
                  "lines.linewidth": 1,
                  "pdf.fonttype": 42,
                  "ps.fonttype": 42
              })

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.tight_layout()

def ax_config(axis):
    axis.set_xlabel('Used Representation', weight="bold")
    axis.set_ylabel('MSE', weight="bold")
    axis.set_title("Normalized Datasets", style="italic")
    axis.set_box_aspect(1)


datasets = ['ASN', 'CCPP', 'CS', 'EEC', 'PT']
to_plot = ['evaluation']

def plot(datasets: list, to_plot: list):
    for dataset in datasets:
        for item in to_plot:
            PATH = "C:\\Users\\thoma\projects\Studium\Bachelorarbeit\suprb-experimentation\logging_output_scripts\outputs\\test\\"
            PATH_COMPLETE = f"{PATH}{dataset}"
            EXT = "*.csv"
            all_csv_files = [file
                             for path, subdir, files in os.walk(PATH_COMPLETE)
                             for file in glob(os.path.join(path, EXT))]

            dataList = []
            for file in all_csv_files:
                if item in file:
                    temp = pd.read_csv(file)
                    dataList.append(temp)
                else:
                    print('nope')

            result = pd.concat(dataList, axis=1)
            temp = result.mul(-1)

            fig, ax = plt.subplots()
            ax = sns.violinplot(data=temp)
            ax_config(ax)
            fig.savefig(f"{PATH_COMPLETE}/{dataset}_eval_plot.png")


plot(datasets, to_plot)