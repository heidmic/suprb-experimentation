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

datasets = ['ASN', 'CCPP', 'CS', 'EEC', 'PT']
to_plot = ['error', 'complexity']

def plot(datasets: list, to_plot: list):
    for dataset in datasets:
        for item in to_plot:
            PATH = "C:\\Users\\thoma\projects\Studium\Bachelorarbeit\suprb-experimentation\logging_output_scripts\outputs\BA Ergebnisse Bereinigt\\"
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

            ax = result.plot(kind='line')
            fig = ax.get_figure()
            fig.savefig(f"{PATH_COMPLETE}/{dataset}_{item}_plot.png")


plot(datasets, to_plot)