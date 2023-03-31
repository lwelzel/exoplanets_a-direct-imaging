import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

FULL_EXOPLANET_ARCHIVE_DATA_FILE = "PS_2023.03.27_03.10.19.csv"

full_wanted_planet_simple_cols = [
    "pl_name",
    "pl_orbsmax",
    "pl_massj",
    "pl_eqt",
    "pl_radj",
    "discoverymethod",
]

full_data = pd.read_csv(FULL_EXOPLANET_ARCHIVE_DATA_FILE, header=290, index_col=0,
                        usecols=full_wanted_planet_simple_cols)

# discovery = [print(e[-1]) for e in full_data.groupby(full_data.index)]


reduced_full_data = full_data.groupby(full_data.index).median()  # MEDIAN

discovery = [e[-1]["discoverymethod"].unique()[0] for e in full_data.groupby(full_data.index)]
reduced_full_data["discoverymethod"] = discovery

gray_orange = sns.color_palette("dark:orange", 2)
reduced_full_data["method_alt"] = [d if d == "Imaging" else "Other" for d in discovery]

reduced_full_data.dropna(axis=0, how="all")

# print(reduced_full_data["discoverymethod"].unique())
# ['Radial Velocity' 'Imaging' 'Eclipse Timing Variations' 'Transit'
#  'Astrometry' 'Disk Kinematics' 'Orbital Brightness Modulation'
#  'Pulsation Timing Variations' 'Microlensing' 'Transit Timing Variations'
#  'Pulsar Timing']

markers = ["."] * len(reduced_full_data["discoverymethod"].unique())
markers[1] = "v"

reduced_full_data.rename(columns={
    'pl_name': 'Name',
    'pl_orbsmax': 'Semi_Major Axis [$AU$]',
    'pl_massj': 'Mass [$M_J$]',
    'pl_eqt': 'Equ. T. [$K$]',
    'pl_radj': 'Radius [$R_J$]',
    'discoverymethod': 'Method',
},
    inplace=True,
)

full_pg = sns.pairplot(reduced_full_data,
                       hue="Method", palette="husl",
                       kind="scatter", diag_kind="kde",
                       corner=True,
                       markers=markers, plot_kws=dict(
        alpha=0.8,
    ),
                       )

axes = full_pg.fig.axes

for ax in axes:
    # print(ax.get_ylabel(), ax.get_xlabel())
    ax.set_xscale("log")
    ax.set_yscale("log")
    # if ax.label
    if ax.get_ylabel() == "Density":
        x_label = ax.get_xlabel()
        _data = [ax.get_xlabel(), "Method"]
        ax.clear()
        sns.histplot(data=reduced_full_data, ax=ax,
                     x=x_label,
                     stat="density", common_norm=False, bins=10, kde=True, multiple="stack",  # multiple="fill",
                     hue="method_alt", palette=gray_orange,
                     log_scale=True,
                     legend=False)
        # sns.kdeplot(data=reduced_full_data, ax=ax,
        #              x=x_label,
        #              common_norm=False,
        #              hue="Method", palette="husl",
        #              log_scale=True,
        #              legend=False)
        ax.yaxis.label.set_visible(False)
        ax.tick_params(axis="both",
                       bottom=True, top=False, left=False, right=False,
                       labelbottom=False, labeltop=False, labelleft=False, labelright=False)


plt.savefig("corner_simple_full.png", dpi=350)
plt.show()
