"""
Grouped boxplots
================

_thumb: .66, .45

"""
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import numpy as np
import matplotlib as mpl
fmri = sns.load_dataset("fmri")
from scipy.stats import sem
sns.set(style="whitegrid")
sns.set_style("ticks")

#############################################################
#Bland-Altman Plot
def bland_altman_plot(data1, data2, *args, **kwargs):
    data1     = np.asarray(data1)
    data2     = np.asarray(data2)
    mean      = np.mean([data1, data2], axis=0)
    diff      = data1 - data2                   # Difference between data1 and data2
    md        = np.mean(diff)                   # Mean of the difference
    sd        = np.std(diff, axis=0, ddof=1)            # Standard deviation of the difference

    plt.scatter(mean, diff, *args, **kwargs)
    plt.axhline(md,           color='gray', linestyle='--')
    plt.axhline(md + 1.96*sd, color='gray', linestyle='--')
    plt.axhline(md - 1.96*sd, color='gray', linestyle='--')
    plt.xlabel('Mean(ms)')
    plt.ylabel('DeltaT1(ms)')

Data=pd.read_csv('C:\Users\\310304075\Desktop\DataPlots\Regression1.csv')
bland_altman_plot(Data['SpinEcho(ms)'], Data['Proposed(ms)'])
plt.title('Bland-Altman Plot')
plt.show()

############################################################
#Linear regression
paper_rc = {'lines.linewidth': 2, 'lines.markersize': 7}
sns.set_context("paper", font_scale=1.5 , rc = paper_rc)

Data=pd.read_csv('C:\Users\\310304075\Desktop\DataPlots\Regression11.csv')

# get coeffs of linear fit
slope, intercept, r_value, p_value, std_err = stats.linregress(Data['IR-SE(ms)'],Data['Proposed(ms)'])
ax=sns.regplot(x="IR-SE(ms)", y="Proposed(ms)", data=Data, color ='black' ,line_kws={'label':"y={0:.3f}x+{1:.3f}".format(slope,intercept)})
ax=sns.regplot(x="IR-SE(ms)", y="Proposed(ms)", data=Data, color ='black' ,line_kws={'label':"y={0:.3f}x+{1:.3f}".format(0.97,23)})
ax=sns.regplot(x="IR-SE(ms)", y="Proposed(ms)", data=Data, color ='black' ,line_kws={'label':"y={0:.3f}x+{1:.3f}".format(1,1.29)})
ax.legend()
#plt.title("Linear Regression")

plt.show()

#############################################################
#Relative Error (Accuracy) Plot

#sns.set(style="white")

tips = sns.load_dataset("tips", True)
Data=pd.read_csv('C:\Users\\310304075\Desktop\DataPlots\RelErrorOurs.csv')

Data['RR(bpm)'] = Data['RR(bpm)'].astype(str)
paper_rc = {'lines.linewidth': 2, 'lines.markersize': 7}
sns.set_context("paper", font_scale=1.5 , rc = paper_rc)
g=sns.lineplot(x="IR-SE(ms)", y="Relative Error",
             hue="HR(bpm)",
             data=Data, palette=sns.color_palette("bright", 8),marker="o")

g.set(ylim=(-0.02, 0.05))
plt.title("Relative Error - Proposed")

plt.show()

#############################################################
#Coefficient of Variation (Precision) Plot

#sns.set(style="white")

tips = sns.load_dataset("tips", True)
Data=pd.read_csv('C:\Users\\310304075\Desktop\DataPlots\CoefVariationSASHA.csv')
Data['RR(bpm)'] = Data['RR(bpm)'].astype(str)

# get coeffs of linear fit
sns.catplot(x="IR-SE(ms)", y="Coefficient of Variation",
             hue="HR(bpm)",
             data=Data, palette=sns.color_palette("muted", 8), kind="bar")
plt.title("Coefficient of Variation - Proposed")

plt.show()

############################################################################################
#Bar plots with std lines

Data=pd.read_csv('C:\Users\\310304075\Desktop\DataPlots\PigsPost.csv')
paper_rc = {'lines.linewidth': 2, 'lines.markersize': 7}
sns.set_context("paper", font_scale=1.5 , rc = paper_rc)
def grouped_barplot(df, cat,subcat, val , err):
    u = df[cat].unique()
    x = np.arange(len(u))
    subx = df[subcat].unique()
    offsets = ((np.arange(len(subx))-np.arange(len(subx)).mean())/(len(subx)+1.))*1.2
    width= np.diff(offsets).mean()
    for i,gr in enumerate(subx):
        dfg = df[df[subcat] == gr]
        plt.bar(x+offsets[i], dfg[val].values, width=width,
                label="{} {}".format(subcat, gr), yerr=dfg[err].values)
    plt.xlabel(cat)
    plt.ylabel(val)
    plt.xticks(x, u)
    plt.legend()
    plt.show()

df=Data
cat = "Swines"
subcat = "Sequence"
val = "T1(ms)"
err = "Std"
grouped_barplot(df, cat, subcat, val, err )

################################################################
#Bull-eyes plot

def bullseye_plot(ax, data, seg_bold=None, cmap=None, norm=None):
    """
    Bullseye representation for the left ventricle.

    Parameters
    ----------
    ax : axes
    data : list of int and float
        The intensity values for each of the 17 segments
    seg_bold : list of int, optional
        A list with the segments to highlight
    cmap : ColorMap or None, optional
        Optional argument to set the desired colormap
    norm : Normalize or None, optional
        Optional argument to normalize data into the [0.0, 1.0] range


    Notes
    -----
    This function create the 17 segment model for the left ventricle according
    to the American Heart Association (AHA) [1]_

    References
    From: https://matplotlib.org/1.5.3/mpl_examples/pylab_examples/leftventricle_bulleye.py
    ----------
    .. [1] M. D. Cerqueira, N. J. Weissman, V. Dilsizian, A. K. Jacobs,
        S. Kaul, W. K. Laskey, D. J. Pennell, J. A. Rumberger, T. Ryan,
        and M. S. Verani, "Standardized myocardial segmentation and
        nomenclature for tomographic imaging of the heart",
        Circulation, vol. 105, no. 4, pp. 539-542, 2002.
    """
    if seg_bold is None:
        seg_bold = []

    linewidth = 2
    data = np.array(data).ravel()

    if cmap is None:
        cmap = plt.cm.viridis

    if norm is None:
        norm = mpl.colors.Normalize(vmin=data.min(), vmax=data.max())

    theta = np.linspace(0, 2 * np.pi, 768)
    r = np.linspace(0.2, 1, 4)

    # Create the bound for the segment 17
    for i in range(r.shape[0]):
        ax.plot(theta, np.repeat(r[i], theta.shape), '-k', lw=linewidth)

    # Create the bounds for the segments 1-12
    for i in range(6):
        theta_i = np.deg2rad(i * 60)
        ax.plot([theta_i, theta_i], [r[1], 1], '-k', lw=linewidth)

    # Create the bounds for the segments 13-16
    for i in range(4):
        theta_i = np.deg2rad(i * 90 - 45)
        ax.plot([theta_i, theta_i], [r[0], r[1]], '-k', lw=linewidth)

    # Fill the segments 1-6
    r0 = r[2:4]
    r0 = np.repeat(r0[:, np.newaxis], 128, axis=1).T
    for i in range(6):
        # First segment start at 60 degrees
        theta0 = theta[i * 128:i * 128 + 128] + np.deg2rad(60)
        theta0 = np.repeat(theta0[:, np.newaxis], 2, axis=1)
        z = np.ones((128, 2)) * data[i]
        ax.pcolormesh(theta0, r0, z, cmap=cmap, norm=norm)
        if i + 1 in seg_bold:
            ax.plot(theta0, r0, '-k', lw=linewidth + 2)
            ax.plot(theta0[0], [r[2], r[3]], '-k', lw=linewidth + 1)
            ax.plot(theta0[-1], [r[2], r[3]], '-k', lw=linewidth + 1)

    # Fill the segments 7-12
    r0 = r[1:3]
    r0 = np.repeat(r0[:, np.newaxis], 128, axis=1).T
    for i in range(6):
        # First segment start at 60 degrees
        theta0 = theta[i * 128:i * 128 + 128] + np.deg2rad(60)
        theta0 = np.repeat(theta0[:, np.newaxis], 2, axis=1)
        z = np.ones((128, 2)) * data[i + 6]
        ax.pcolormesh(theta0, r0, z, cmap=cmap, norm=norm)
        if i + 7 in seg_bold:
            ax.plot(theta0, r0, '-k', lw=linewidth + 2)
            ax.plot(theta0[0], [r[1], r[2]], '-k', lw=linewidth + 1)
            ax.plot(theta0[-1], [r[1], r[2]], '-k', lw=linewidth + 1)

    # Fill the segments 13-16
    r0 = r[0:2]
    r0 = np.repeat(r0[:, np.newaxis], 192, axis=1).T
    for i in range(4):
        # First segment start at 45 degrees
        theta0 = theta[i * 192:i * 192 + 192] + np.deg2rad(45)
        theta0 = np.repeat(theta0[:, np.newaxis], 2, axis=1)
        z = np.ones((192, 2)) * data[i + 12]
        ax.pcolormesh(theta0, r0, z, cmap=cmap, norm=norm)
        if i + 13 in seg_bold:
            ax.plot(theta0, r0, '-k', lw=linewidth + 2)
            ax.plot(theta0[0], [r[0], r[1]], '-k', lw=linewidth + 1)
            ax.plot(theta0[-1], [r[0], r[1]], '-k', lw=linewidth + 1)

    # Fill the segments 17
    if data.size == 17:
        r0 = np.array([0, r[0]])
        r0 = np.repeat(r0[:, np.newaxis], theta.size, axis=1).T
        theta0 = np.repeat(theta[:, np.newaxis], 2, axis=1)
        z = np.ones((theta.size, 2)) * data[16]
        ax.pcolormesh(theta0, r0, z, cmap=cmap, norm=norm)
        if 17 in seg_bold:
            ax.plot(theta0, r0, '-k', lw=linewidth + 2)

    ax.set_ylim([0, 1])
    ax.set_yticklabels([])
    ax.set_xticklabels([])


data=pd.read_csv('C:\Users\\310304075\Desktop\DataPlots\BullOurs.csv')
data1 = np.asarray(data['Mean'])
data2 = np.asarray(data['Std'])

fig, ax = plt.subplots(figsize=(12, 8), nrows=1, ncols=2,
                       subplot_kw=dict(projection='polar'))
fig.canvas.set_window_title('T1 myocardium')

# Create the axis for the colorbars
axl = fig.add_axes([0.20, 0.15, 0.2, 0.05])
axl2 = fig.add_axes([0.625, 0.15, 0.2, 0.05])

# Set the colormap and norm to correspond to the data for which
# the colorbar will be used.
cmap = mpl.cm.viridis_r
norm = mpl.colors.Normalize(vmin=1350, vmax=2350)

# ColorbarBase derives from ScalarMappable and puts a colorbar
# in a specified axes, so it has everything needed for a
# standalone colorbar.  There are many more kwargs, but the
# following gives a basic continuous colorbar with ticks
# and labels.
cb1 = mpl.colorbar.ColorbarBase(axl, cmap=cmap, norm=norm,
                                orientation='horizontal')
cb1.set_label('T1 (ms)')


# Set the colormap and norm to correspond to the data for which
# the colorbar will be used.
cmap2 = mpl.cm.viridis_r
norm2 = mpl.colors.Normalize(vmin=0, vmax=0.08)

# ColorbarBase derives from ScalarMappable and puts a colorbar
# in a specified axes, so it has everything needed for a
# standalone colorbar.  There are many more kwargs, but the
# following gives a basic continuous colorbar with ticks
# and labels.
cb2 = mpl.colorbar.ColorbarBase(axl2, cmap=cmap2, norm=norm2,
                               orientation='horizontal')
cb2.set_label('Std (ms)')
cb2.set_label('CV')

# Create the 17 segment model
bullseye_plot(ax[0], data1, cmap=cmap, norm=norm)
ax[0].set_title('T1 Values myocardium')

bullseye_plot(ax[1], data2, cmap=cmap2, norm=norm2)
ax[1].set_title('Standard deviation myocardium')
ax[1].set_title('Coefficient of Variation')

plt.show()

