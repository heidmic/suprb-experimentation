from logging_output_scripts.utils import check_and_create_dir, get_dataframe, get_all_runs, get_df
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from sklearn.preprocessing import MinMaxScaler

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

gheDf = pd.read_csv('C:\\Users\\thoma\projects\Studium\Bachelorarbeit\suprb-experimentation\logging_output_scripts\outputs\\test\gheTest.csv')
heDf = pd.read_csv('C:\\Users\\thoma\projects\Studium\Bachelorarbeit\suprb-experimentation\logging_output_scripts\outputs\\test\HE_CS.csv')
obrDf = pd.read_csv('C:\\Users\\thoma\projects\Studium\Bachelorarbeit\suprb-experimentation\logging_output_scripts\outputs\\test\obrTest.csv')

result = pd.concat([gheDf, heDf, obrDf], axis=1)
test = result.mul(1)
ax = test.plot(kind='line')
#plt.show()
fig = ax.get_figure()
fig.savefig("C:\\Users\\thoma\projects\Studium\Bachelorarbeit\suprb-experimentation\logging_output_scripts/test.png")
print('hi')
def ax_config(axis):
    ax.set_xlabel('Used Representation', weight="bold")
    ax.set_ylabel('MSE', weight="bold")
    ax.set_title("Normalized Datasets", style="italic")
    ax.set_box_aspect(1)

HEDF = pd.read_csv('C:\\Users\\thoma\projects\Studium\Bachelorarbeit\suprb-experimentation\logging_output_scripts\\HE.csv')
HEDF2 = pd.read_csv('C:\\Users\\thoma\projects\Studium\Bachelorarbeit\suprb-experimentation\logging_output_scripts\\HE2.csv')

result2 = pd.concat([HEDF, HEDF2], axis=1)
test2 = result2.mul(-1)

fig, ax = plt.subplots()
ax = sns.violinplot(data=test2)
ax_config(ax)
fig.savefig("C:\\Users\\thoma\projects\Studium\Bachelorarbeit\suprb-experimentation\logging_output_scripts/2.png")