"""Generate nice plots for paper."""

import os
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from matplotlib.ticker import MaxNLocator

plt.rcParams.update({'font.size': 22})

import wandb
api = wandb.Api()
entity = 'pby'

from itertools import cycle

MARKERSIZE = 10
LINEWIDTH = 4

# ============================================================================
# Utils
# ============================================================================

def smooth_data(scalars, weight=0.):
    last = scalars[0]
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return np.array(smoothed)

def tsplot(data, x=None, smooth=0., marker=None, label=None, **kw):
    if x is None:
        x = np.arange(data.shape[0])
    # Plot data's smoothed mean
    y = np.mean(data, axis=1)
    y = smooth_data(y, weight=smooth)
    # Find standard deviation and error
    sd = np.std(data, axis=1)
    se = sd/np.sqrt(data.shape[1])
    # Plot
    plt.plot(x, y, marker=marker, markersize=MARKERSIZE, linewidth=LINEWIDTH, label=label, **kw)
    # Show error on graph
    cis = (y-sd, y+sd)
    plt.fill_between(x, cis[0], cis[1], alpha=0.2, **kw)

def plot_legend(legends, colors, markers, save_name):
    # Dummy plots
    for legend, color, marker in zip(legends, colors, markers):
        plt.plot([0,0,0], [0,0,0], color=color, label=legend, marker=marker, markersize=MARKERSIZE, linewidth=LINEWIDTH)
    # Get legend separately
    handles, labels = plt.gca().get_legend_handles_labels()
    leg = plt.legend(handles, labels, loc='center', ncol=len(legends))
    plt.axis('off')
    fig = leg.figure
    fig.canvas.draw()
    bbox = leg.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(save_name, bbox_inches=bbox, pad_inches=0, dpi=500)
    plt.close('all')


def plot_horizontal_colorbar_legend(vmin, vmax, cmap, label, save_name):
    norm = Normalize(vmin=vmin, vmax=vmax)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    fig, ax = plt.subplots(figsize=(4, 0.2))
    cb = fig.colorbar(sm, cax=ax, orientation='horizontal')
    # cb.set_label(label, fontsize=10)
    cb.ax.tick_params(labelsize=5)

    sm.set_cmap(cmap)
    fig.savefig(save_name, bbox_inches='tight', pad_inches=0.1, dpi=500)
    plt.close(fig)


# ============================================================================
# Main plotting
# ============================================================================

def retrieve_group(project, group, metric, x_axis, prepend=None):
    # Get runs
    path = os.path.join(entity, project)
    runs = api.runs(path=path, filters={"config.group": group})
    # Get data
    data = [run.history()[metric] for run in runs]
    min_length = min([d.shape[0] for d in data])
    data = np.concatenate([datum.to_numpy()[:min_length,None] for datum in data], axis=-1)
    # Just get x-axis of one run since all runs should be identical
    x_axis = runs[0].history()[x_axis].to_numpy()[:min_length]

    # Filter out nans
    c_data, c_x_axis = [], []
    for datum, x in zip(data, x_axis):
        # if (datum == 'NaN').any() == False:
        if np.sum(np.isnan(datum)) == 0:
            c_data += [datum]
            c_x_axis += [x]
        # elif group == 'PointCircle-ML-final':
        #     c_data += [np.zeros_like(datum)]
        #     c_x_axis += [x]


    data, x_axis = np.array(c_data), np.array(c_x_axis)
    if prepend is not None:
        data, x_axis = prepend_missing_points(data, x_axis, prepend)
    return data[::2], x_axis[::2]

def prepend_missing_points(data, x_axis, points):
    x_axis = np.concatenate([[0], x_axis])
    points = points[:data.shape[1]]
    points = np.reshape(points, [1,data.shape[1]])
    data = np.concatenate([points, data], axis=0)
    return data, x_axis

def plot(data, x_axis=None, min_x_axis=None, smooth=0., legend=None, color=None, marker=None):
    if x_axis is not None and min_x_axis is not None:
        # Take evenly spaced points to match mix_x_axis
        r = int(x_axis.shape[0]/min_x_axis.shape[0])
        indices = np.arange(0, x_axis.shape[0], r)
        #indices = list(filter(lambda idx: x_axis[idx] >= min_x_axis[0], indices))
        x_axis = x_axis[indices]
        data = data[indices]

    tsplot(data, x=x_axis, smooth=smooth, marker=marker, label=legend, color=color)

def plot_graph(project, groups, metrics, x_axes, save_name, xlim=None, ylim=None, legends=None, smooth=0.,
               colors=None, markers=None, horizontal_lines=None, horizontal_lines_colors=None, horizontal_lines_legends=None,
               horizontal_lines_markers=None, ylabel_length=None, prepend=None, x_label=None, y_label=None, correct_x_axis=False,
               show_legend=False, n_plot_every=1):
    plt.figure(figsize=(6, 4))
    # Retrieve data
    metrics = [metrics]*len(groups) if type(metrics) != list else metrics
    x_axes = [x_axes]*len(groups) if type(x_axes) != list else x_axes
    data = [retrieve_group(project, *args, prepend=prepend) for args in zip(groups, metrics, x_axes)]

    # Take value at equally spaced intervals
    min_x_axis = min([x_axis for _, x_axis in data], key=lambda x: x.shape[0])

    # Plot any horizontal lines
    if horizontal_lines is not None:
        hcolors = [horizontal_lines_colors]*len(groups) if type(horizontal_lines_colors) != list else horizontal_lines_colors
        hmarkers = [horizontal_lines_markers]*len(groups) if type(horizontal_lines_markers) != list else horizontal_lines_markers
        for line, color, legend, marker in zip(horizontal_lines, hcolors, horizontal_lines_legends, hmarkers):
            plt.plot(min_x_axis, line*np.ones(min_x_axis.shape), linewidth=LINEWIDTH, marker=marker, markersize=MARKERSIZE, color=color, label=legend)

    # Plot data
    legends = [legends]*len(groups) if type(legends) != list else legends
    colors = [colors]*len(groups) if type(colors) != list else colors
    markers = [markers]*len(groups) if type(markers) != list else markers
    for (datum, x_axis), legend, color, marker in zip(data, legends, colors, markers):
        plot(datum[::n_plot_every,:], x_axis[::n_plot_every], min_x_axis[::n_plot_every], smooth, legend, color, marker)

    # Format plot
    if xlim is not None:
        plt.xlim(*xlim)
    if ylim is not None:
        plt.ylim(*ylim)
    if plt.yticks()[0][-1] >= 2000:
        ylabels = ['%d' % y + 'k' for y in plt.yticks()[0]/1000]
        plt.gca().set_yticklabels(ylabels)

    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))

    # Correct ylabels length (will cast to int)
    if ylabel_length is not None:
        ax = plt.gca()
        ylabels = [' '*(ylabel_length-len(str(int(y))))+'%d'%y for y in plt.yticks()[0]]
        ax.set_yticklabels(ylabels)
    plt.margins(x=0)
    plt.gca().grid(which='major', linestyle='-', linewidth=1, color='gray')  # Increase linewidth and change color
    plt.grid('on')

    # Label axes
    if x_label is not None:
        plt.xlabel(x_label)
    if y_label is not None:
        plt.title(y_label)

    if show_legend:
        plt.legend(loc='upper left', prop={'size': 12})

    # Save
    #plt.show()
    plt.savefig(save_name, bbox_inches='tight', pad_inches=0, dpi=500)

    plt.close()

# ============================================================================
# What to plot?
# ============================================================================

def main_results(save_dir):
    project = 'ICRL-FE2'
    smooth = 0.
    #colors = ['r', '#006400', '#adadad', '#00008b', '#ff8c00']
    #colors = ['r', '#006400', 'y', '#fed8b1', '#add8e6']
    colors = ['r', '#006400', 'y', '#9932a8', '#1f5fc4']
    #markers = ['s', '^', 'p', '*', 'X']
    markers = [None, None, None, None, None]

    # ========================================================================
    # Legends
    # ========================================================================

    plot_legend(['PUCL (ours)', 'MECL',  'GPUCL', 'BC'], colors, markers, os.path.join(save_dir, 'legend.png'))
    # plot_legend(['ours', 'MCEL'], colors, markers, os.path.join(save_dir, 'legend_no_expert.png'))

    # ========================================================================
    # Learning constraints
    # ========================================================================

    def pointellip_lc():
        sd = os.path.join(save_dir, 'pointellip')
        os.makedirs(sd, exist_ok=True)

        # Data
        point_reward_at_zero = None
        # np.array([
        #     0,
        # ])
        point_cost_at_zero = None
        # np.array([
        #     0.0,
        # ])
        point_nominal_reward = 0.02
        point_expert_reward = 0.0931
        point_nominal_cost = 0.
        point_expert_cost = 0.

        # point Reward
        # plot_graph(
        #     project=project,
        #     groups=['PointCircle-BCE-final','PointCircle-ML-final'],
        #     metrics=['true/average_reward', 'true/average_reward'],
        #     x_axes=['timesteps','timesteps'],
        #     save_name=os.path.join(sd, 'cir_reward.png'),
        #     # xlim=[0, 3.85e6],
        #     # ylim=[0, 0.1],
        #     legends=[None, None, None],
        #     smooth=smooth,
        #     colors=colors[:3],
        #     markers=markers[:3],
        #     # horizontal_lines=[point_expert_reward],
        #     horizontal_lines_colors=colors[3:],
        #     horizontal_lines_markers=markers[:3],
        #     horizontal_lines_legends=[None, None],
        #     prepend=point_reward_at_zero,
        #     x_label='timesteps',
        #     y_label='Reward',
        # )

        # point violation
        plot_graph(
            project=project,
            groups=['2d_pu','2d_ml_new5','2d_gpu', '2d_bce'],
            metrics=['true/violation_steps_portion', 'true/violation_steps_portion', 'true/violation_steps_portion', 'true/violation_steps_portion'],
            x_axes=['timesteps','timesteps','timesteps','timesteps'],
            save_name=os.path.join(sd, 'pointellip_violation.png'),
            xlim=[0, 1.2e6],
            ylim=[0, 0.32],
            legends=[None, None, None, None],
            smooth=smooth,
            colors=colors[:4],
            markers=markers[:4],
            # horizontal_lines=[point_expert_cost],
            horizontal_lines_colors=colors[3:],
            horizontal_lines_markers=markers[:3],
            horizontal_lines_legends=[None, None],
            prepend=point_cost_at_zero,
            x_label='timesteps',
            y_label='Unsafe Rate - 2D reach',
        )

        plot_graph(
            project=project,
            groups=['2d_pu','2d_ml_new5','2d_gpu', '2d_bce'],
            metrics=['true/jaccard', 'true/jaccard', 'true/jaccard', 'true/jaccard'],
            x_axes=['timesteps','timesteps','timesteps','timesteps'],
            save_name=os.path.join(sd, 'pointellip_jaccard.png'),
            xlim=[0, 1.2e6],
            ylim=[0, 0.8],
            legends=[None, None, None, None],
            smooth=smooth,
            colors=colors[:4],
            markers=markers[:4],
            horizontal_lines=[],
            horizontal_lines_colors=colors[3:],
            horizontal_lines_markers=markers[:3],
            horizontal_lines_legends=[None, None],
            prepend=point_cost_at_zero,
            x_label='timesteps',
            y_label='IoU - 2D reach',
        )

    def reach_lc():
        sd = os.path.join(save_dir, 'reachobs')
        os.makedirs(sd, exist_ok=True)

        # Data
        point_reward_at_zero = None
        # np.array([
        #     0,
        # ])
        point_cost_at_zero = None
        # np.array([
        #     0.0,
        # ])
        point_nominal_reward = 0.02
        point_expert_reward = 0.0931
        point_nominal_cost = 0.
        point_expert_cost = 0.

        # point Reward
        # plot_graph(
        #     project=project,
        #     groups=['PointCircle-BCE-final','PointCircle-ML-final'],
        #     metrics=['true/average_reward', 'true/average_reward'],
        #     x_axes=['timesteps','timesteps'],
        #     save_name=os.path.join(sd, 'cir_reward.png'),
        #     # xlim=[0, 3.85e6],
        #     # ylim=[0, 0.1],
        #     legends=[None, None, None],
        #     smooth=smooth,
        #     colors=colors[:3],
        #     markers=markers[:3],
        #     # horizontal_lines=[point_expert_reward],
        #     horizontal_lines_colors=colors[3:],
        #     horizontal_lines_markers=markers[:3],
        #     horizontal_lines_legends=[None, None],
        #     prepend=point_reward_at_zero,
        #     x_label='timesteps',
        #     y_label='Reward',
        # )

        # point violation
        plot_graph(
            project=project,
            groups=['3d_pu', '3d_ml', '3d_gpu', '3d_bc'],
            metrics=['true/violation_steps_portion', 'true/violation_steps_portion', 'true/violation_steps_portion',
                     'true/violation_steps_portion'],
            x_axes=['timesteps', 'timesteps', 'timesteps', 'timesteps'],
            save_name=os.path.join(sd, 'reachobs_violation.png'),
            xlim=[0, 0.9e6],
            ylim=[0, 0.35],
            legends=[None, None, None, None],
            smooth=smooth,
            colors=colors[:4],
            markers=markers[:4],
            # horizontal_lines=[point_expert_cost],
            horizontal_lines_colors=colors[3:],
            horizontal_lines_markers=markers[:3],
            horizontal_lines_legends=[None, None],
            prepend=point_cost_at_zero,
            x_label='timesteps',
            y_label='Unsafe rate - 3D reach',
        )

        plot_graph(
            project=project,
            groups=['3d_pu', '3d_ml', '3d_gpu', '3d_bc'],
            metrics=['true/jaccard', 'true/jaccard', 'true/jaccard', 'true/jaccard'],
            x_axes=['timesteps', 'timesteps', 'timesteps', 'timesteps'],
            save_name=os.path.join(sd, 'reachobs_jaccard.png'),
            xlim=[0, .9e6],
            ylim=[0, 0.6],
            legends=[None, None, None, None],
            smooth=smooth,
            colors=colors[:4],
            markers=markers[:4],
            horizontal_lines=[],
            horizontal_lines_colors=colors[3:],
            horizontal_lines_markers=markers[:3],
            horizontal_lines_legends=[None, None],
            prepend=point_cost_at_zero,
            x_label='timesteps',
            y_label='IoU - 3D reach',
        )

    def halfcheetah_lc():
        sd = os.path.join(save_dir, 'halfcheetah')
        os.makedirs(sd, exist_ok=True)

        # Data
        point_reward_at_zero = None
        # np.array([
        #     0,
        # ])
        point_cost_at_zero = None
        # np.array([
        #     0.0,
        # ])
        # point_nominal_reward = 0.02
        # point_expert_reward = 0.0931
        # point_nominal_cost = 0.
        # point_expert_cost = 0.

        # point Reward
        # plot_graph(
        #     project=project,
        #     groups=['PointCircle-BCE-final','PointCircle-ML-final'],
        #     metrics=['true/average_reward', 'true/average_reward'],
        #     x_axes=['timesteps','timesteps'],
        #     save_name=os.path.join(sd, 'cir_reward.png'),
        #     # xlim=[0, 3.85e6],
        #     # ylim=[0, 0.1],
        #     legends=[None, None, None],
        #     smooth=smooth,
        #     colors=colors[:3],
        #     markers=markers[:3],
        #     # horizontal_lines=[point_expert_reward],
        #     horizontal_lines_colors=colors[3:],
        #     horizontal_lines_markers=markers[:3],
        #     horizontal_lines_legends=[None, None],
        #     prepend=point_reward_at_zero,
        #     x_label='timesteps',
        #     y_label='Reward',
        # )

        # point violation
        plot_graph(
            project=project,
            groups=['HC_pu', 'HC_ml', 'HC_gpu', 'HC_bce'],
            metrics=['true/violation_steps_portion', 'true/violation_steps_portion', 'true/violation_steps_portion',
                     'true/violation_steps_portion'],
            x_axes=['timesteps', 'timesteps', 'timesteps', 'timesteps'],
            save_name=os.path.join(sd, 'halfcheetah_violation.png'),
            xlim=[0, 1.53e6],
            ylim=[0, 1],
            legends=[None, None, None, None],
            smooth=smooth,
            colors=colors[:4],
            markers=markers[:4],
            # horizontal_lines=[point_expert_cost],
            horizontal_lines_colors=colors[3:],
            horizontal_lines_markers=markers[:3],
            horizontal_lines_legends=[None, None],
            prepend=point_cost_at_zero,
            x_label='timesteps',
            y_label='Unsafe rate - blocked half cheetah',
            n_plot_every=2
        )

        # plot_graph(
        #     project=project,
        #     groups=['HC_pu', 'HC_ml', 'HC_gpu', 'HC_bce'],
        #     metrics=['true/jaccard', 'true/jaccard', 'true/jaccard', 'true/jaccard'],
        #     x_axes=['timesteps', 'timesteps', 'timesteps', 'timesteps'],
        #     save_name=os.path.join(sd, 'halfcheetah_jaccard.png'),
        #     xlim=[0, 1.53e6],
        #     ylim=[0, 1],
        #     legends=[None, None, None, None],
        #     smooth=smooth,
        #     colors=colors[:4],
        #     markers=markers[:4],
        #     horizontal_lines=[],
        #     horizontal_lines_colors=colors[3:],
        #     horizontal_lines_markers=markers[:3],
        #     horizontal_lines_legends=[None, None],
        #     prepend=point_cost_at_zero,
        #     x_label='timesteps',
        #     y_label='IoU - blocked half cheetah',
        #     n_plot_every=2
        # )

    def sensitivity_analysis():
        knn_threshold = np.array([0.02, 0.04, 0.08, 0.12, 0.16, 0.2, 0.3])
        unsafe_rate = np.array([7.04, 5.17, 0.41, 1.97, 4.23, 11.67, 20.17])
        iou = np.array([0.316, 0.486, 0.633, 0.758, 0.787, 0.703, 0.436])

        fig, ax1 = plt.subplots(figsize=(5, 2))

        color = 'tab:blue'
        ax1.set_xlabel('Distance threshold $d_r$', fontsize=10)
        ax1.set_ylabel('Unsafe Rate (%)', color=color, fontsize=10)
        ax1.plot(knn_threshold, unsafe_rate, marker='o', color=color, label='Unsafe Rate')
        ax1.tick_params(axis='y', labelcolor=color, labelsize=10)
        ax1.tick_params(axis='x', labelsize=10)
        ax1.set_ylim([0, 35])
        ax1.yaxis.set_major_locator(MaxNLocator(nbins=4))

        ax2 = ax1.twinx()
        color = 'tab:orange'
        ax2.set_ylabel('IoU', color=color, fontsize=10)
        ax2.plot(knn_threshold, iou, marker='s', linestyle='--', color=color, label='IoU')
        ax2.tick_params(axis='y', labelcolor=color, labelsize=10)
        ax2.set_ylim([0, 1])
        ax2.yaxis.set_major_locator(MaxNLocator(nbins=4))

        fig.tight_layout()
        ax1.grid(True)
        plt.savefig(os.path.join(save_dir, 'sensitivity.png'), bbox_inches='tight', pad_inches=0, dpi=500)

        # plt.title('Sensitivity Analysis of kNN Threshold')
        plt.show()

    # ========================================================================
    # Make all plots
    # ========================================================================

    pointellip_lc()
    # reach_lc()
    # halfcheetah_lc()
    # sensitivity_analysis()


if __name__=='__main__':
    start = time.time()
    main_results('icrl/plots/main_results')
    # ablation_studies('plots/ablations')
    print('Time taken: ', (time.time()-start))
