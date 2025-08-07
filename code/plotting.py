import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
import colorsys

tab10 = plt.rcParams['axes.prop_cycle'].by_key()['color']
FUNCTION_COLOR   = tab10[0]
BEST_POLY_COLOR  = "black"
DPP_COLOR        = tab10[3]
LEV_COLOR        = tab10[9]

def darker(color, amount=0.7):
    r, g, b, a = color
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    r2, g2, b2 = colorsys.hls_to_rgb(h, max(0, l*amount), s)
    return (r2, g2, b2, a)
    
def darker2(color, amount=0.7): #patch for last plot...
    r, g, b, a = mcolors.to_rgba(color)
    h, l, s    = colorsys.rgb_to_hls(r, g, b)
    r2, g2, b2 = colorsys.hls_to_rgb(h, max(0, l * amount), s)
    return (r2, g2, b2, a)   
    
def quant_stats(eps):
    med  = np.median(eps, axis=1)
    low  = np.percentile(eps, 10, axis=1)
    high = np.percentile(eps, 90, axis=1)
    return med, low, high
    
def _plot_res1_panel(ax,
                     res1,
                     label_fontsize=12,
                     tick_fontsize=10,
                     line_thickness=2,
                     show_ylabel=True):
    FC = globals().get('FUNCTION_COLOR',  'black')
    PC = globals().get('BEST_POLY_COLOR', 'gray')
    DC = globals().get('DPP_COLOR',       'tab:red')
    LC = globals().get('LEV_COLOR',       'tab:blue')

    grid       = res1["grid"]
    true_y     = res1["true_y"]
    best_y     = res1["best_y"]
    mean_u     = res1["mean_unb"]
    std_u      = res1["std_unb"]
    mean_l     = res1["mean_bia"]
    std_l      = res1["std_bia"]
    interval   = res1.get("interval", None)

    l_f,  = ax.plot(grid, true_y, lw=line_thickness+1, color=FC, label='f')
    l_p,  = ax.plot(grid, best_y, lw=line_thickness+1, color=PC, label='$p^*$')

    l_deb, = ax.plot(grid, mean_u, '--', lw=line_thickness,
                     color=DC, label='Debiased sampling')
    ax.fill_between(grid, mean_u - std_u, mean_u + std_u,
                    color=DC, alpha=0.15)

    l_lev, = ax.plot(grid, mean_l, '--', lw=line_thickness,
                     color=LC, label='Leverage score sampling')
    ax.fill_between(grid, mean_l - std_l, mean_l + std_l,
                    color=LC, alpha=0.15)

    if interval is not None:
        ax.set_xlim(*interval)

    ax.set_xlabel('t', fontsize=label_fontsize)
    if show_ylabel:
        ax.set_ylabel('f(t)', fontsize=label_fontsize)
    ax.tick_params(labelsize=tick_fontsize)
    ax.set_ylim(-3, 3)

    try:
        ax.set_box_aspect(1)          
    except Exception:
        ax.set_aspect('equal', adjustable='box')

    return [l_f, l_p, l_deb, l_lev]
    
def plot_figure1(res1_a,
                       res1_b,
                       title_a=None,
                       title_b=None,
                       save_path=None,
                       figsize=(12, 6),
                       dpi=300,
                       label_fontsize=12,
                       tick_fontsize=12,
                       legend_fontsize=13,
                       line_thickness=2,
                       wspace=0.4,
                       hspace=0.0):
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=figsize, dpi=dpi)

    handles_left = _plot_res1_panel(ax_left, res1_a,
                                    label_fontsize=label_fontsize,
                                    tick_fontsize=tick_fontsize,
                                    line_thickness=line_thickness,
                                    show_ylabel=True)

    _plot_res1_panel(ax_right, res1_b,
                     label_fontsize=label_fontsize,
                     tick_fontsize=tick_fontsize,
                     line_thickness=line_thickness,
                     show_ylabel=True)

    if title_a is not None:
        ax_left.set_title(title_a, fontsize=label_fontsize+2, pad=8)
    if title_b is not None:
        ax_right.set_title(title_b, fontsize=label_fontsize+2, pad=8)

    if res1_a.get("interval") is not None:
        a0, a1 = res1_a["interval"]
    else:
        a0, a1 = ax_left.get_xlim()
    ax_left.set_xticks(np.arange(np.floor(a0), np.ceil(a1) + 0.5, 0.5))

    labels = [h.get_label() for h in handles_left]
    fig.legend(handles_left, labels,
               loc='upper center',
               bbox_to_anchor=(0.5, .95),
               ncol=2,
               fontsize=legend_fontsize,
               frameon=True)

    fig.subplots_adjust(wspace=wspace, hspace=hspace)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
    plt.show()

def plot_figure2(
    res1,
    res4,
    save_path=None,
    label_fontsize=12,
    legend_fontsize=10,
    tick_fontsize=10,
    line_thickness=2
):
    theta     = res1["theta"]
    true_y    = res1["true_y"]
    best_y    = res1["best_y"]
    mean_u, std_u = res1["mean_unb"], res1["std_unb"]
    mean_l, std_l = res1["mean_bia"], res1["std_bia"]

    results   = res4["results"]
    d_values  = res4["d_values"]

    pi_ticks  = np.pi * np.array([0, 0.5, 1.0, 1.5, 2.0])
    pi_labels = [r'$0$', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$']

    palette = plt.cm.winter
    num_colors     = len(d_values)
    zeta_colors = palette(np.linspace(0.4, 0.8, num_colors))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

    ax1.plot(theta, true_y,
             label='f',
             lw=line_thickness + 1,
             color=FUNCTION_COLOR)
    ax1.plot(theta, best_y,
             label=r'$p^*$',
             lw=line_thickness + 1,
             color=BEST_POLY_COLOR)
    ax1.plot(theta, mean_u, '--',
             label='Debiased Sampling',
             color=DPP_COLOR,
             lw=line_thickness)
    ax1.fill_between(theta, mean_u - std_u, mean_u + std_u,
                     color=DPP_COLOR, alpha=0.15)
    ax1.plot(theta, mean_l, '--',
             label='Leverage score sampling',
             color=LEV_COLOR,
             lw=line_thickness)
    ax1.fill_between(theta, mean_l - std_l, mean_l + std_l,
                     color=LEV_COLOR, alpha=0.15)

    ax1.set_xlim(0, 2*np.pi)
    ax1.set_ylim(-3, 3)
    ax1.set_xticks(pi_ticks)
    ax1.set_xticklabels(pi_labels, fontsize=tick_fontsize)
    ax1.set_ylabel(r'Re $f(e^{i\theta})$', fontsize=label_fontsize)
    ax1.tick_params(direction='in', labelsize=tick_fontsize)
    ax1.set_xlabel(r"$\theta$", fontsize=label_fontsize)


    ax1.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, 1.25),
        bbox_transform=ax1.transAxes,
        ncol=2,
        fontsize=legend_fontsize,
        frameon=True,
        fancybox=True,
        handlelength=2,
        columnspacing=1.0
    ).get_frame().set_edgecolor('0.3')


    for idx, d in enumerate(d_values):
        n_vals       = results[d]["n_values"]
        eps_u, eps_l = results[d]["eps_debiased"], results[d]["eps_leverage"]
        med_u, low_u, high_u = quant_stats(eps_u)
        med_l, low_l, high_l = quant_stats(eps_l)
    
        base_color = zeta_colors[idx]
        edge_color = darker(base_color)
    
        ax2.plot(
            n_vals, med_u,
            label=f"$d={d}$",
            color=edge_color,
            marker='o',
            markerfacecolor=base_color,
            markeredgecolor=edge_color,
            markeredgewidth=1.2,
            linestyle='-',
            lw=line_thickness
        )
        ax2.fill_between(n_vals, low_u, high_u,
                         facecolor=base_color, alpha=0.3)
    
        ax2.plot(
            n_vals, med_l,
            label=f"$d={d}$",
            color=edge_color,
            marker='^',
            markerfacecolor=base_color,
            markeredgecolor=edge_color,
            markeredgewidth=1.2,
            linestyle='--',
            lw=line_thickness
        )
        ax2.fill_between(n_vals, low_l, high_l,
                         facecolor=base_color, alpha=0.15)

    ax2.set_yscale("log")
    ax2.set_xlabel("Number of samples $n$", fontsize=label_fontsize)
    ax2.set_ylabel(
        r"$\varepsilon = \frac{\|f - \hat p\|_{L^2}}{\|f - p^*\|_{L^2}} - 1$",
        fontsize=label_fontsize
    )
    ax2.tick_params(labelsize=tick_fontsize)

    ax2.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, 1.25),
        bbox_transform=ax2.transAxes,
        ncol=3,
        fontsize=legend_fontsize,
        frameon=True,
        fancybox=True
    ).get_frame().set_edgecolor('0.3')

    fig.subplots_adjust(wspace=0.3, top=0.80)

    # if save_path:
    plt.savefig("fourier_final",dpi=300, bbox_inches='tight')
    plt.show()




def plot_figure3(
    res4_unif,
    res4_gauss,
    save_path=None,
    label_fontsize=12,
    tick_fontsize=9,
    legend_fontsize=10,
    line_thickness=2,
    wspace=0.4,
    hspace=0.4
):
    tab10     = plt.rcParams['axes.prop_cycle'].by_key()['color']
    DPP_COLOR = tab10[3]
    LEV_COLOR = tab10[9]
    
    d_all  = res4_unif["d_values"]
    d_vals = [d_all[0], d_all[-1]]
    
    fig, axes = plt.subplots(
        1, 4,
        figsize=(16, 4),
        dpi=300
    )

    plot_configs = [
        (d_vals[0], res4_unif, 'Uniform'),
        (d_vals[1], res4_unif, 'Uniform'),
        (d_vals[0], res4_gauss, 'Gaussian'),
        (d_vals[1], res4_gauss, 'Gaussian')
    ]
    
    for idx, (d, res4, measure) in enumerate(plot_configs):
        ax = axes[idx]
        ax.grid(True)
        
        data    = res4["results"][d]
        n_vals  = data["n_values"]
        eps_u   = data["eps_debiased"]
        eps_l   = data["eps_leverage"]
        med_u, lo_u, hi_u = quant_stats(eps_u)
        med_l, lo_l, hi_l = quant_stats(eps_l)

        edge_u = darker2(DPP_COLOR)
        ax.plot(n_vals, med_u,
                marker='o', linestyle='-',
                color=edge_u, markerfacecolor=DPP_COLOR,
                markeredgecolor=edge_u, lw=line_thickness)
        ax.fill_between(n_vals, lo_u, hi_u, facecolor=DPP_COLOR, alpha=0.3)

        edge_l = darker2(LEV_COLOR)
        ax.plot(n_vals, med_l,
                marker='^', linestyle='-',
                color=edge_l, markerfacecolor=LEV_COLOR,
                markeredgecolor=edge_l, lw=line_thickness)
        ax.fill_between(n_vals, lo_l, hi_l, facecolor=LEV_COLOR, alpha=0.15)

        ax.set_yscale('log')
        ax.set_box_aspect(1)
        ax.set_xlabel('Sample complexity $n$', fontsize=label_fontsize)
        ax.set_ylabel(r'$\varepsilon_{\mathrm{empirical}}$', fontsize=label_fontsize+2)
        ax.tick_params(labelsize=tick_fontsize)
        ax.set_title(f'{measure} – degree {d}', fontsize=label_fontsize)

    handle_u = Line2D([], [], marker='o', linestyle='-',
                      color=darker2(DPP_COLOR), markerfacecolor=DPP_COLOR,
                      label='Debiased sampling', lw=line_thickness)
    handle_l = Line2D([], [], marker='^', linestyle='-',
                      color=darker2(LEV_COLOR), markerfacecolor=LEV_COLOR,
                      label='Leverage score sampling', lw=line_thickness)

    fig.legend(
        [handle_u, handle_l],
        ['Debiased sampling', 'Leverage score sampling'],
        loc='upper center',
        bbox_to_anchor=(0.5, 1.01),
        bbox_transform=fig.transFigure,
        ncol=2,
        fontsize=legend_fontsize,
        frameon=True,
        fancybox=True
    ).get_frame().set_edgecolor('0.3')

    fig.subplots_adjust(top=0.88, wspace=wspace, hspace=hspace)
    
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()
