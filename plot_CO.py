import matplotlib as mpl
import matplotlib.pyplot as plt
plt.rcParams['figure.constrained_layout.use'] = True
import pandas as pd
import numpy as np
import seaborn as sns

from labellines import labelLine, labelLines

FULL_EXOPLANET_CO_FILE = "DIP_CO_ratio.csv"

data = pd.read_csv(FULL_EXOPLANET_CO_FILE, header=0)



sorting = np.argsort(data['C/O'].to_numpy())

data = data.reindex(sorting)

# data["Name"].rename(index={
#     'kappa And b': '$\kappa$ And b',
# },)

data.loc[data["Name"] =='kappa And b', "Name"] = '$\kappa$ And b'

cmap = mpl.colormaps['viridis']
seps = data["separation (au)"].to_numpy().astype(float)
# seps = np.log10(data["separation (au)"].to_numpy().astype(float))
# seps = (seps - np.min(seps)) / (np.max(seps) - np.min(seps))
# seps = (seps - np.min(seps)) / (np.max(seps) - np.min(seps))
hue = cmap(seps / 300.)

hue_dict = {s:c for s, c in zip(seps, hue)}

CO_min, CO_mean, CO_max = (-data['C/O low']).to_numpy(), data['C/O'].to_numpy(), (data['C/O up']).to_numpy()
CO_all = np.array([CO_mean - CO_min, CO_mean, CO_mean + CO_max])
yerr = np.array([CO_min, CO_max])

print(data.columns)
print(data["Temperature K"])

# fig, ax = plt.subplots(1, 1, figsize=(5,5))
#
# ax.bar(data=data, label="Name", x="C/O",
#        color=hue,
#        xerr=yerr,
#        )

ax = sns.barplot(data, y="Name", x="C/O",
                 hue=seps,  # order=sorting,
                 xerr=yerr,  # capsize=14., errwidth=1.,
                 palette=hue_dict,
                 width=0.9,
                 dodge=False,
                 )

for _artist in ax.lines + ax.collections + ax.patches + ax.images:
    _artist.set_label(s=None)

ax.errorbar(data["C/O"], data["Name"],  xerr=yerr,
             fmt='D', markersize=4, capsize=6, color="r")

ax.get_legend().remove()

fig = plt.gcf()

cax = fig.add_axes([0.95, 0.22, 0.025, 0.62])

norm = mpl.colors.Normalize(vmin=0., vmax=300.)
sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
sm.set_array([])
cb = plt.colorbar(sm, cax=cax, orientation='vertical',
                               extend='both',
                               # label="$M_{core}$ [$M_{\oplus}$]",
                               # ticks=[0, 3, 6, 9],
                  )
cax.yaxis.set_ticks_position('left')
cax.yaxis.set_label_position('left')
ticklabs = cb.ax.get_yticklabels()
cb.ax.set_yticklabels(ticklabs, fontsize=6)
cb.set_label("Separation [AU]", fontsize=8)


# ax.axvline(1., color="gray", ls="dashed")

hj = ax.axvline(0.35, color="gray", ls="solid", label="HJ (Fleury+2020)")
oc_trans = ax.axvline(0.9, color="gray", ls="dashed", label=r"O$\rightarrow $C chem. transition (Polman+2022)")
haze = ax.axvline(0.625, color="dimgray", ls="-.", label=r"$\varepsilon_{haze} \uparrow \forall \forall \lambda$ (Corrales+2023)")

mean = np.mean(CO_all)
std = np.std(CO_all)

m = ax.axvline(mean, color="darkblue", ls="dotted", label="Mean TW")

# ax.fill_betweenx(["51 Eri b", "HIP 65426 b"],
#                  mean - std,
#                  mean + std,
#                  color="lightblue", alpha=0.4, zorder=0)

ax.fill_between(np.linspace(0, 1., 100), 0, 1, where=np.logical_and((mean - std) < np.linspace(0, 1., 100),
                                                                   np.linspace(0, 1., 100) < (mean + std)),
                transform=ax.get_xaxis_transform(),
                color="lightblue", alpha=0.4, zorder=0
                )


ax.yaxis.label.set_visible(False)

ax.legend([hj,
           m,
           ],
          ["Hot. Jup. (Fleury+2020)",
           "Sample Mean",],
          bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
          mode="expand", borderaxespad=0, ncol=2,
          fontsize=8,
          )

bbox_white = {
    "boxstyle": "Round, pad=0.15",
    "edgecolor": "none",
    "facecolor": "white",
    "linewidth": 0.2,
}
labelLines(lines=[oc_trans,
                  haze],
           labels=[r"O$\rightarrow$C rich atm. (Polman+2022)",
                   r"$\varepsilon_{haze} \uparrow \forall \forall \lambda$ (Corrales+2023)"],
           align=True,
           fontsize=7,
           outline_color="none",
           backgroundcolor="white",
           bbox=bbox_white,
           zorder=99,
           ha="left")

colorsbg = np.ones(len(data["Temperature K"]))
colorsbg[0] = 0

for teff, name, bg in zip(data["Temperature K"], data["Name"], colorsbg):
    bbox = None if bg==1 else bbox_white
    c = "white" if bg == 1 else "black"
    ax.annotate(text=f"{teff:.0f} K",
                xy=(0.15, name),
                color=c, bbox=bbox,
                ha='right', va="center")

ax.set_xlim(None, 1.1)
ax.annotate(text=f"$1\sigma$ CI",
            xy=(0.99 * (mean + std), "HIP 65426 b"),
            ha='right', va="bottom")

s = ""
for index, row in data.iterrows():
    s = s + f"{row['Name']}: {row['references']} & {row['Source abundances, presence']} & {row['Source Temperature']}, "
    # print(row['Name'], row['Name'])

print(s)

plt.savefig("CO_ratio_direct_imaging.png", dpi=350)
plt.show()

