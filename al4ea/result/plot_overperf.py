# -*- coding: utf-8 -*-





import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn import metrics
import numpy as np
import pandas as pd
import itertools
import os
import json
from scipy.stats import ttest_ind, ttest_rel
from statsmodels.stats.multitest import multipletests


def load_result(fn):
    res_df = pd.read_csv(fn, header=0, index_col=0)
    hit1 = res_df.loc["hit@1"]
    hit1 = hit1.to_numpy()
    return hit1


def draw_overall_performance(res_dir):
    data_name_list = ["D_W_15K_V1", "EN_DE_15K_V1"]
    bach_percent_list = [0.0, 0.3]
    ea_model_list = ["BootEA", "Alinet", "RDGCN"]
    strategy_name_list = ["RAND", "DEGREE", "PAGERANK", "BETWEEN", "UNCERTAINTY", "STRUCT_UNCER_0.1", "STRUCT_UNCER_BACH_RECOG_CV_0.1"]

    model_num = 3
    dt_num = 4
    fig, axes = plt.subplots(model_num, dt_num, figsize=(25, 15))

    marker_list = ["*", "^", "o", ">", "s", "D", "v"]
    colors = sbn.color_palette()

    anno_percent_list = [0.00, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
    ylims = [[(0, 100), (0, 100)], [(0, 100), (0, 100)], [(30, 80), (65, 95)]]

    strategy_name_in_fig_list = ["rand", "degree", "pagerank", "betweenness", "uncertainty", "struct_uncert", "ActiveEA"]
    data_name_in_fig_list = ["DW", "ENDE"]

    for m_idx in range(len(ea_model_list)):
        ea_model = ea_model_list[m_idx]
        for d_idx in range(len(data_name_list)):
            data_name = data_name_list[d_idx]
            for p_idx in range(len(bach_percent_list)):
                percent = bach_percent_list[p_idx]
                ax = axes[m_idx, 2*d_idx+p_idx]
                for s_idx in range(len(strategy_name_list)):
                    strategy = strategy_name_list[s_idx]
                    dir_name = f"{data_name}_BACH{percent}_{ea_model}"
                    fp = os.path.join(res_dir, dir_name, f"{strategy}.csv")
                    if not os.path.exists(fp):
                        continue
                    hit1 = load_result(fp)
                    ax.plot(anno_percent_list, hit1, label=strategy_name_in_fig_list[s_idx], color=colors[s_idx], marker=marker_list[s_idx])
                ax.set_title(f"{ea_model}, {data_name_in_fig_list[d_idx]} ({int(100*percent)}%)", fontsize=20)
                ax.set_ylim(*ylims[m_idx][d_idx])
                bot, top = ylims[m_idx][d_idx]
                if m_idx == 2:
                    ax.set_yticks(np.arange(bot, top, 5))
                else:
                    ax.set_yticks(np.arange(bot, top, 10))
                ax.grid()
                ax.tick_params(axis="x", labelsize=15)
                ax.tick_params(axis="y", labelsize=15)
    lines, labels = axes[1, 1].get_legend_handles_labels()
    fig.legend(lines, labels, loc="lower center", ncol=7, fontsize=23, bbox_to_anchor=(0.5, -0.05))
    axes[2, 1].set_xlabel("Percentage of Annotated Entities", fontsize=25)
    axes[2, 1].xaxis.set_label_coords(1.1, -0.15)
    axes[1, 0].set_ylabel("Hit@1 (%)", fontsize=25)
    plt.margins(0, 0)
    fig.tight_layout()
    plt.savefig(os.path.join(fig_save_dir, "overall_perf.png"), pad_inches=0, bbox_inches="tight")
    fig.show()


def draw_overall_performance_2x6(res_dir):
    data_name_list = ["D_W_15K_V1", "EN_DE_15K_V1"]
    bach_percent_list = [0.0, 0.3]
    ea_model_list = ["BootEA", "Alinet", "RDGCN"]
    strategy_name_list = ["RAND", "DEGREE", "PAGERANK", "BETWEEN", "UNCERTAINTY", "STRUCT_UNCER_0.1", "STRUCT_UNCER_BACH_RECOG_CV_0.1"]

    model_num = 3
    dt_num = 2
    fig, axes = plt.subplots(len(bach_percent_list), model_num*dt_num, figsize=(30, 10))

    marker_list = ["*", "^", "o", ">", "s", "D", "v"]
    colors = sbn.color_palette()

    anno_percent_list = [0.00, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
    ylims = [[(0, 100), (0, 100)], [(0, 100), (0, 100)], [(0, 100), (0, 100)]]

    strategy_name_in_fig_list = ["rand", "degree", "pagerank", "betweenness", "uncertainty", "struct_uncert", "ActiveEA"]
    data_name_in_fig_list = ["DW", "ENDE"]

    for p_idx in range(len(bach_percent_list)):
        percent = bach_percent_list[p_idx]
        for m_idx in range(len(ea_model_list)):
            ea_model = ea_model_list[m_idx]
            for d_idx in range(len(data_name_list)):
                data_name = data_name_list[d_idx]

                ax = axes[p_idx, 2*m_idx+d_idx]
                for s_idx in range(len(strategy_name_list)):
                    strategy = strategy_name_list[s_idx]
                    dir_name = f"{data_name}_BACH{percent}_{ea_model}"
                    fp = os.path.join(res_dir, dir_name, f"{strategy}.csv")
                    if not os.path.exists(fp):
                        continue
                    hit1 = load_result(fp)
                    ax.plot(anno_percent_list, hit1, label=strategy_name_in_fig_list[s_idx], color=colors[s_idx], marker=marker_list[s_idx])
                ax.set_title(f"{ea_model}, {data_name_in_fig_list[d_idx]} ({int(100*percent)}%)", fontsize=20)
                ax.set_ylim(*ylims[m_idx][d_idx])
                # bot, top = ylims[m_idx][d_idx]
                ax.set_yticks(np.arange(0, 100, 10))
                ax.grid()
                ax.tick_params(axis="x", labelsize=15)
                ax.tick_params(axis="y", labelsize=15)
                ax.set_xlim([-0.05, 0.55])
    lines, labels = axes[1, 1].get_legend_handles_labels()
    fig.legend(lines, labels, loc="lower center", ncol=7, fontsize=26, bbox_to_anchor=(0.5, -0.08))
    axes[1, 2].set_xlabel("Percentage of Annotated Entities", fontsize=25)
    # axes[1, 2].xaxis.set_label_coords(1.1, -0.15)
    axes[1, 0].set_ylabel("Hit@1 (%)", fontsize=25)
    axes[0, 0].set_ylabel("Hit@1 (%)", fontsize=25)
    plt.margins(0, 0)
    fig.tight_layout()
    plt.savefig(os.path.join(fig_save_dir, "overall_perf_2x6.pdf"), pad_inches=0, bbox_inches="tight")
    fig.show()



def draw_effect_of_mc(res_dir):
    data_name_list = ["D_W_15K_V1", "EN_DE_15K_V1"]
    bach_percent_list = [0.0, 0.3]
    ea_model_list = ["Alinet", "RDGCN"]
    uncert_strategy_name_list = ["UNCERTAINTY", "STRUCT_UNCER_0.1", "STRUCT_UNCER_BACH_RECOG_CV_0.1"]
    deep_uncert_strategy_name_list = ["DEEP_UNCERTAINTY", "DEEP_STRUCT_UNCER_0.1", "DEEP_STRUCT_UNCER_BACH_RECOG_CV_0.1"]

    uncert_strategy_name_in_fig_list = ["uncertainty", "struct_uncert", "ActiveEA"]
    deep_uncert_strategy_name_in_fig_list = ["uncertainty + Bayesian", "struct_uncert + Bayesian", "ActiveEA + Bayesian"]
    data_name_in_fig_list = ["DW", "ENDE"]

    model_num = 2
    dt_num = 4
    fig, axes = plt.subplots(model_num, dt_num, figsize=(25, 10))

    marker_list = ["*", "^", "o", ">", "s", "D", "v"]
    colors = sbn.color_palette()

    anno_percent_list = [0.00, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
    ylims = [[(0, 100), (0, 100)], [(30, 100), (60, 100)]]

    for p_idx in range(len(bach_percent_list)):
        for m_idx in range(len(ea_model_list)):
            ea_model = ea_model_list[m_idx]
            for d_idx in range(len(data_name_list)):
                data_name = data_name_list[d_idx]
                percent = bach_percent_list[p_idx]
                ax = axes[m_idx, 2 * d_idx + p_idx]
                for s_idx in range(len(uncert_strategy_name_list)):
                    uncert_strategy = uncert_strategy_name_list[s_idx]
                    deep_uncert_strategy = deep_uncert_strategy_name_list[s_idx]
                    dir_name = f"{data_name}_BACH{percent}_{ea_model}"
                    uncert_fp = os.path.join(res_dir, dir_name, f"{uncert_strategy}.csv")
                    deep_uncert_fp = os.path.join(res_dir, dir_name, f"{deep_uncert_strategy}.csv")
                    if not os.path.exists(uncert_fp):
                        continue
                    uncert_hit1 = load_result(uncert_fp)
                    deep_uncert_hit1 = load_result(deep_uncert_fp)
                    ax.plot(anno_percent_list, uncert_hit1, label=uncert_strategy_name_in_fig_list[s_idx], color=colors[s_idx], linestyle="solid")
                    ax.plot(anno_percent_list, deep_uncert_hit1, label=deep_uncert_strategy_name_in_fig_list[s_idx], color=colors[s_idx], linestyle="dotted")
                ax.set_title(f"{ea_model}, {data_name_in_fig_list[d_idx]} ({percent})", fontsize=20)
                ax.set_ylim(*ylims[m_idx][d_idx])
                bot, top = ylims[m_idx][d_idx]
                if m_idx == 2:
                    ax.set_yticks(np.arange(bot, top, 5))
                else:
                    ax.set_yticks(np.arange(bot, top, 10))
                ax.grid()
                ax.tick_params(axis="x", labelsize=15)
                ax.tick_params(axis="y", labelsize=15)

    lines, labels = axes[1, 1].get_legend_handles_labels()
    fig.legend(lines, labels, loc="lower center", ncol=6, fontsize=22.5, bbox_to_anchor=(0.5, -0.1))
    axes[1, 1].set_xlabel("Percentage of Annotated Entities", fontsize=25)
    axes[1, 1].xaxis.set_label_coords(1.1, -0.15)
    axes[1, 0].set_ylabel("Hit@1 (%)", fontsize=25)
    # axes[1, 0].yaxis.set_label_coords(-0.1, 1.4)
    axes[0, 0].set_ylabel("Hit@1 (%)", fontsize=25)
    # axes[0, 0].yaxis.set_label_coords(-0.1, 1.4)

    fig.tight_layout()
    plt.savefig(os.path.join(fig_save_dir, "effect_of_mc.png"), pad_inches=0, bbox_inches="tight")
    fig.show()


def draw_effect_of_bayesian2(res_dir, overall_res_dir):
    data_name_list = ["D_W_15K_V1", "EN_DE_15K_V1"]
    bach_percent_list = [0.0, 0.3]
    ea_model_list = ["Alinet", "RDGCN"]

    uncert_strategy_name_list = ["UNCERTAINTY", "STRUCT_UNCER_0.1", "STRUCT_UNCER_BACH_RECOG_CV_0.1"]
    deep_uncert_strategy_name_list = ["DEEP_UNCERTAINTY_dropout{}", "DEEP_STRUCT_UNCER_alpha0.1_batchsize100_dropout{}", "DEEP_STRUCT_UNCER_BACH_RECOG_CV_alpha0.1_batchsize100_dropout{}"]

    uncert_strategy_name_in_fig_list = ["uncertainty", "ActiveEA"]
    deep_uncert_strategy_name_in_fig_list = ["uncertainty + Bayesian", "struct_uncert + Bayesian",
                                             "ActiveEA + Bayesian"]
    data_name_in_fig_list = ["DW", "ENDE"]

    marker_list = ["*", "^", "o", ">", "s", "D", "v"]
    colors = sbn.color_palette()

    fig_num = 3
    anno_percent_list = [0.00, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
    fig, axes = plt.subplots(1, fig_num, figsize=(10, 5))
    dropout_rates = [0.05, 0.1, 0.2]

    for drop_idx in range(fig_num):
        dropout = dropout_rates[drop_idx]
        ax = axes[drop_idx]
        query_list = []
        uncert_auc_list = []
        deep_uncert_auc_list = []
        color_list = []
        for p_idx in range(len(bach_percent_list)):
            for m_idx in range(len(ea_model_list)):
                ea_model = ea_model_list[m_idx]
                for d_idx in range(len(data_name_list)):
                    data_name = data_name_list[d_idx]
                    percent = bach_percent_list[p_idx]
                    for s_idx in range(len(uncert_strategy_name_in_fig_list)):
                        if s_idx == 1 and percent > 0:
                            uncert_strategy = uncert_strategy_name_list[s_idx + 1]
                            deep_uncert_strategy = deep_uncert_strategy_name_list[s_idx + 1].format(dropout)
                        else:
                            uncert_strategy = uncert_strategy_name_list[s_idx]
                            deep_uncert_strategy = deep_uncert_strategy_name_list[s_idx].format(dropout)


                        dir_name = f"{data_name}_BACH{percent}_{ea_model}"
                        uncert_fp = os.path.join(overall_res_dir, dir_name, f"{uncert_strategy}.csv")
                        deep_uncert_fp = os.path.join(res_dir, dir_name, f"{deep_uncert_strategy}.csv")
                        if not os.path.exists(deep_uncert_fp):
                            continue
                        uncert_hit1 = load_result(uncert_fp)
                        deep_uncert_hit1 = load_result(deep_uncert_fp)
                        uncert_auc = metrics.auc(anno_percent_list, uncert_hit1)
                        deep_uncert_auc = metrics.auc(anno_percent_list, deep_uncert_hit1)
                        uncert_auc_list.append(uncert_auc)
                        deep_uncert_auc_list.append(deep_uncert_auc)
                        query_list.append(f"{uncert_strategy_name_in_fig_list[s_idx]},{ea_model},{data_name_in_fig_list[d_idx]}({int(100*percent)}%)")
                        color_list.append(colors[s_idx])

        diff_arr = np.array(deep_uncert_auc_list) - np.array(uncert_auc_list)
        sort_idxes = np.argsort(diff_arr)

        sorted_query_list = np.array(query_list)[sort_idxes]
        sorted_color_list = np.array(color_list)[sort_idxes]
        sorted_diff_list = diff_arr[sort_idxes]
        for idx in range(len(sort_idxes)):
            ax.barh(idx, sorted_diff_list[idx], color=sorted_color_list[idx], label="x")

        ax.tick_params(axis="x", labelsize=15)
        ax.set_title("dropout=%s"%str(dropout), fontsize=25)
        ax.set_yticks([])

    axes[1].set_xlabel("Difference of AUC@0.5", fontsize=25)
    axes[0].set_ylabel("Runs", fontsize=25)
    lines, labels = axes[1].get_legend_handles_labels()
    # fig.legend(lines, labels, loc="lower left", ncol=4, fontsize=15, bbox_to_anchor=(0.1, -0.15))
    axes[1].legend([lines[0], lines[1]], ["uncertainty", "ActiveEA"], fontsize=12)
    fig.tight_layout()
    plt.savefig(os.path.join(fig_save_dir, "effect_of_bayesian.pdf"), pad_inches=0, bbox_inches="tight")
    plt.show()




def calculte_overall_performance_auc(res_dir):

    data_name_list = ["D_W_15K_V1", "EN_DE_15K_V1"]
    bach_percent_list = [0.0, 0.3]
    ea_model_list = ["BootEA", "Alinet", "RDGCN"]
    strategy_name_list = ["RAND", "DEGREE", "PAGERANK", "BETWEEN", "UNCERTAINTY", "STRUCT_UNCER_0.1",
                          "STRUCT_UNCER_BACH_RECOG_CV_0.1"]


    anno_percent_list = [0.00, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
    ylims = [[(0, 100), (0, 100)], [(0, 100), (0, 100)], [(30, 80), (65, 95)]]

    strategy_name_in_fig_list = ["rand", "degree", "pagerank", "betweenness", "uncertainty", "ActiveEA (struct_uncert)",
                                 "ActiveEA (full)"]
    data_name_in_fig_list = ["DW", "ENDE"]

    for s_idx in range(len(strategy_name_list)):
        line_results = []
        for m_idx in range(len(ea_model_list)):
            ea_model = ea_model_list[m_idx]
            for d_idx in range(len(data_name_list)):
                data_name = data_name_list[d_idx]
                for p_idx in range(len(bach_percent_list)):
                    percent = bach_percent_list[p_idx]

                    strategy = strategy_name_list[s_idx]
                    dir_name = f"{data_name}_BACH{percent}_{ea_model}"
                    fp = os.path.join(res_dir, dir_name, f"{strategy}.csv")
                    if not os.path.exists(fp):
                        line_results.append("-")
                        continue
                    hit1 = load_result(fp)
                    auc = metrics.auc(anno_percent_list, hit1)
                    line_results.append("%.1f"%auc)
        print(strategy_name_in_fig_list[s_idx] + " & " + " & ".join(line_results) + " \\\\")


def print_overall_performance_auc_table(out_dir):
    data_name_list = ["D_W_15K_V1", "EN_DE_15K_V1"]
    bach_percent_list = [0.0, 0.3]
    ea_model_list = ["BootEA", "Alinet", "RDGCN"]
    strategy_name_list = ["RAND", "DEGREE", "PAGERANK", "BETWEEN", "UNCERTAINTY", "STRUCT_UNCER_alpha0.1_batchsize100",
                          "STRUCT_UNCER_BACH_RECOG_CV_alpha0.1_batchsize100"]

    anno_percent_list = [0.00, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
    ylims = [[(0, 100), (0, 100)], [(0, 100), (0, 100)], [(30, 80), (65, 95)]]

    strategy_name_in_fig_list = ["rand", "degree", "pagerank", "betweenness", "uncertainty",
                                 "struct_uncert", "ActiveEA"]
    data_name_in_fig_list = ["DW", "ENDE"]

    results_map = dict()
    for s_idx in range(len(strategy_name_list)):
        for m_idx in range(len(ea_model_list)):
            ea_model = ea_model_list[m_idx]
            for d_idx in range(len(data_name_list)):
                data_name = data_name_list[d_idx]
                for p_idx in range(len(bach_percent_list)):
                    percent = bach_percent_list[p_idx]
                    strategy = strategy_name_list[s_idx]
                    dir_name = f"{data_name}_BACH{percent}_{ea_model}"
                    auc_list = []
                    for seed in [1011, 1012, 1013, 1014, 1015]:

                        fp = os.path.join(f"{out_dir}/{seed}/results/overall_perf/", dir_name, f"{strategy}.csv")
                        if not os.path.exists(fp):
                            if percent == 0 and strategy=="STRUCT_UNCER_BACH_RECOG_CV_alpha0.1_batchsize100":
                                continue
                            print("not found: ", fp)
                            continue
                        hit1 = load_result(fp)
                        auc = metrics.auc(anno_percent_list, hit1)
                        auc_list.append(auc)
                    results_map[(ea_model, data_name, percent, strategy)] = auc_list

    mean_line_list = []
    p_line_list = []
    for s_idx in range(len(strategy_name_list)):
        line_results = []
        p_results = []
        for m_idx in range(len(ea_model_list)):
            ea_model = ea_model_list[m_idx]
            for d_idx in range(len(data_name_list)):
                data_name = data_name_list[d_idx]
                for p_idx in range(len(bach_percent_list)):
                    percent = bach_percent_list[p_idx]
                    strategy = strategy_name_list[s_idx]
                    k = (ea_model, data_name, percent, strategy)
                    if k in results_map and len(results_map[k]):
                        line_results.append("%.1f" % np.mean(results_map[k]))
                        sa_k = (ea_model, data_name, percent, strategy_name_list[-2])
                        acea_k = (ea_model, data_name, percent, strategy_name_list[-1])
                        activeea_auc_list = results_map[acea_k] if len(results_map[acea_k]) else results_map[sa_k]
                        # test_res = ttest_rel(activeea_auc_list, results_map[k])
                        test_res = ttest_ind(activeea_auc_list, results_map[k])
                        p_results.append("%.5f" % test_res[1])
                    else:
                        line_results.append("-")
                        p_results.append("-")

        mean_line = strategy_name_in_fig_list[s_idx] + " & " + " & ".join(line_results) + " \\\\"
        p_line = strategy_name_in_fig_list[s_idx] + " & " + " & ".join(p_results) + " \\\\"
        mean_line_list.append(mean_line)
        p_line_list.append(p_line)

    print("== mean value table ==")
    for line in mean_line_list:
        print(line)
    print("== p value table ==")
    for line in p_line_list:
        print(line)

    corrected_p_map = dict()
    for m_idx in range(len(ea_model_list)):
        ea_model = ea_model_list[m_idx]
        for d_idx in range(len(data_name_list)):
            data_name = data_name_list[d_idx]
            for p_idx in range(len(bach_percent_list)):
                percent = bach_percent_list[p_idx]

                sa_k = (ea_model, data_name, percent, strategy_name_list[-2])
                acea_k = (ea_model, data_name, percent, strategy_name_list[-1])
                activeea_auc_list = results_map[acea_k] if len(results_map[acea_k]) else results_map[sa_k]

                p_list = []
                for s_idx in range(len(strategy_name_list)-2):
                    strategy = strategy_name_list[s_idx]
                    k = (ea_model, data_name, percent, strategy)
                    test_res = ttest_rel(activeea_auc_list, results_map[k])
                    p_list.append(test_res[1])
                _, p_corrected, _, _ = multipletests(p_list, method="bonferroni", alpha=0.05)
                corrected_p_map[(ea_model, data_name, percent)] = p_corrected


    correct_p_line_list = []
    for s_idx in range(len(strategy_name_list[:-2])):
        line_results = []
        for m_idx in range(len(ea_model_list)):
            ea_model = ea_model_list[m_idx]
            for d_idx in range(len(data_name_list)):
                data_name = data_name_list[d_idx]
                for p_idx in range(len(bach_percent_list)):
                    percent = bach_percent_list[p_idx]
                    k = (ea_model, data_name, percent)
                    line_results.append("%.5f" % corrected_p_map[k][s_idx])
        cor_p_line = strategy_name_in_fig_list[s_idx] + " & " + " & ".join(line_results) + " \\\\"
        correct_p_line_list.append(cor_p_line)
    print("== corrected p values ==")
    for line in correct_p_line_list:
        print(line)













def draw_overall_performance_on_enfr100k(res_dir):
    data_name = "EN_FR_100K_V1"
    bach_percent_list = [0.0, 0.3]
    ea_model = "Alinet"
    strategy_name_list = ["RAND", "DEGREE", "PAGERANK", "BETWEEN", "UNCERTAINTY", "STRUCT_UNCER_0.1",
                          "STRUCT_UNCER_BACH_RECOG_CV_alpha0.1_batchsize1000"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    marker_list = ["*", "^", "o", ">", "s", "D", "v"]
    colors = sbn.color_palette()

    anno_percent_list = [0.00, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]

    strategy_name_in_fig_list = ["rand", "degree", "pagerank", "betweenness", "uncertainty", "struct_uncert",
                                 "ActiveEA"]


    for p_idx in range(len(bach_percent_list)):
        percent = bach_percent_list[p_idx]
        ax = axes[p_idx]
        for s_idx in range(len(strategy_name_list)):
            strategy = strategy_name_list[s_idx]
            dir_name = f"{data_name}_BACH{percent}_{ea_model}"
            fp = os.path.join(res_dir, dir_name, f"{strategy}.csv")
            if not os.path.exists(fp):
                continue
            hit1 = load_result(fp)
            ax.plot(anno_percent_list, hit1, label=strategy_name_in_fig_list[s_idx], color=colors[s_idx],
                    marker=marker_list[s_idx])
        ax.set_title(f"{ea_model}, ENFR ({int(100*percent)}%)", fontsize=25)
        ax.grid()
        ax.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        ax.tick_params(axis="x", labelsize=15)
        ax.tick_params(axis="y", labelsize=15)
        ax.set_ylim(0, 60)
        ax.set_xlim(-0.05, 0.55)
        ax.set_ylabel("Hit@1 (%)", fontsize=20)
    axes[0].set_xlabel("Percentage of Annotated Entities", fontsize=20)
    axes[0].xaxis.set_label_coords(1.1, -0.15)
    lines, labels = axes[1].get_legend_handles_labels()
    fig.legend(lines, labels, loc="lower left", ncol=4, fontsize=15, bbox_to_anchor=(0.1, -0.15))
    fig.tight_layout()
    plt.savefig(os.path.join(fig_save_dir, "overall_perf_enfr100k.pdf"), pad_inches=0.05, bbox_inches="tight")
    fig.show()




def draw_effect_of_alpha(res_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    marker_list = ["*", "^", "o", ">", "s", "D", "v"]
    colors = sbn.color_palette()

    data_name = "D_W_15K_V1"
    bach_percent_list = ["0.0", "0.3"]
    ea_model_list = ["Alinet", "BootEA"]

    anno_percent_list = [0.00, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
    alpha_range = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


    for m_idx in range(len(ea_model_list)):
        ax = axes[m_idx]
        model_name = ea_model_list[m_idx]
        for p_idx in range(len(bach_percent_list)):
            percent = bach_percent_list[p_idx]
            auc_list1 = []
            for alpha in alpha_range:
                strategy = f"STRUCT_UNCER_{alpha}"
                dir_name = f"{data_name}_BACH{percent}_{model_name}"
                fp = os.path.join(res_dir, dir_name, f"{strategy}.csv")
                hit1 = load_result(fp)
                auc = metrics.auc(anno_percent_list, hit1)
                auc_list1.append(auc)
            ax.plot(alpha_range, auc_list1, label=f"{percent}", color=colors[p_idx], marker=marker_list[p_idx])

            # pagerank
            strategy = f"PAGERANK"
            dir_name = f"{data_name}_BACH{percent}_{model_name}"
            fp = os.path.join(res_dir, dir_name, f"{strategy}.csv")
            hit1 = load_result(fp)
            pr_auc = metrics.auc(anno_percent_list, hit1)
            # uncertainty
            strategy = f"UNCERTAINTY"
            dir_name = f"{data_name}_BACH{percent}_{model_name}"
            fp = os.path.join(res_dir, dir_name, f"{strategy}.csv")
            hit1 = load_result(fp)
            uncert_auc = metrics.auc(anno_percent_list, hit1)

            ax.plot([0, 1], [pr_auc]*2, label="pagerank", color=colors[p_idx], linestyle="dotted")
            ax.text(-0.04, pr_auc-0.4, 'pagerank', horizontalalignment='left', verticalalignment='center', fontsize=15)
            ax.plot([0, 1], [uncert_auc] * 2, label="uncertainty", color=colors[p_idx], linestyle="dotted")
            ax.text(-0.04, uncert_auc-0.4, 'uncertainty', horizontalalignment='left', verticalalignment='center', fontsize=15)


            ax.set_title(f"{model_name} on DW", fontsize=25)
            ax.set_xlabel("alpha", fontsize=20)
            ax.set_ylabel("AUC@0.5", fontsize=20)
            ax.tick_params(axis="x", labelsize=15)
            ax.tick_params(axis="y", labelsize=15)


            # auc_list1 = []
            # for alpha in alpha_range:
            #     strategy = f"STRUCT_UNCER_BACH_RECOG_CV_{alpha}"
            #     dir_name = f"{data_name}_BACH{percent}_{model_name}"
            #     fp = os.path.join(res_dir, dir_name, f"{strategy}.csv")
            #     hit1 = load_result(fp)
            #     auc = metrics.auc(anno_percent_list, hit1)
            #     auc_list1.append(auc)
            # ax.plot(alpha_range, auc_list1, label="ActiveEA (full)", color=colors[1], marker=marker_list[1])

    axes[0].set_ylim(13, 25)
    axes[1].set_ylim(10, 30)


    lines, labels = axes[1].get_legend_handles_labels()
    fig.legend([lines[0], lines[3]], [labels[0], labels[3]], loc="lower center", ncol=2, fontsize=20, bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    plt.savefig(os.path.join(fig_save_dir, "effect_of_alpha.pdf"), pad_inches=0, bbox_inches="tight")
    fig.show()




def draw_effect_of_batchsize():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    marker_list = ["*", "^", "o", ">", "s", "D", "v"]
    colors = sbn.color_palette()

    data_name = "D_W_15K_V1"
    bach_percent_list = [0.0, 0.3]
    ea_model_list = ["Alinet", "BootEA"]

    anno_percent_list = [0.00, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
    batchsize_range = [100, 200, 300, 400, 500, 1000]

    for m_idx in range(len(ea_model_list)):
        model_name = ea_model_list[m_idx]
        ax = axes[m_idx]

        for p_idx in range(len(bach_percent_list)):
            percent = bach_percent_list[p_idx]

            if percent == 0.0:
                auc_list1 = []
                for batchsize in batchsize_range:
                    strategy = f"STRUCT_UNCER_alpha{0.1}_batchsize{batchsize}"
                    dir_name = f"{data_name}_BACH{percent}_{model_name}"
                    fp = os.path.join(res_dir, dir_name, f"{strategy}.csv")
                    hit1 = load_result(fp)
                    auc = metrics.auc(anno_percent_list, hit1)
                    auc_list1.append(auc)
                ax.plot(batchsize_range, auc_list1, label=f"{percent}", color=colors[p_idx], marker=marker_list[p_idx])
            elif percent > 0:
                auc_list1 = []
                for batchsize in batchsize_range:
                    strategy = f"STRUCT_UNCER_BACH_RECOG_CV_alpha{0.1}_batchsize{batchsize}"
                    dir_name = f"{data_name}_BACH{percent}_{model_name}"
                    fp = os.path.join(res_dir, dir_name, f"{strategy}.csv")
                    hit1 = load_result(fp)
                    auc = metrics.auc(anno_percent_list, hit1)
                    auc_list1.append(auc)
                ax.plot(batchsize_range, auc_list1, label=f"{percent}", color=colors[p_idx], marker=marker_list[p_idx])

        ax.set_title(f"{model_name} on DW", fontsize=25)
        ax.set_xlabel("batch size of sampling", fontsize=20)
        ax.set_ylabel("AUC@0.5", fontsize=20)
        ax.tick_params(axis="x", labelsize=15)
        ax.tick_params(axis="y", labelsize=15)


    axes[0].set_ylim(20, 30)
    axes[1].set_ylim(20, 30)


    lines, labels = axes[1].get_legend_handles_labels()
    fig.legend(lines, labels, loc="lower center", ncol=2, fontsize=20, bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    plt.savefig(os.path.join(fig_save_dir, "effect_of_batchsize.png"), pad_inches=0, bbox_inches="tight")
    fig.show()



def draw_sensitivity_of_parameters(res_dir):
    task_group = "effect_of_alpha"  # overall_perf, effect_of_alpha
    task_res_dir = os.path.join(res_dir, task_group)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    marker_list = ["*", "^", "o", ">", "s", "D", "v"]
    colors = sbn.color_palette()

    data_name = "D_W_15K_V1"
    bach_percent_list = [0.0, 0.3]
    ea_model_list = ["Alinet", "BootEA"]
    anno_percent_list = [0.00, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
    alpha_range = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    batchsize_range = [100, 200, 300, 400, 500, 1000]
    for m_idx in range(len(ea_model_list)):
        model_name = ea_model_list[m_idx]
        for p_idx in range(len(bach_percent_list)):
            ax = axes[0, p_idx]
            percent = bach_percent_list[p_idx]
            auc_list1 = []
            for alpha in alpha_range:
                strategy = f"STRUCT_UNCER_{alpha}"
                dir_name = f"{data_name}_BACH{percent}_{model_name}"
                fp = os.path.join(task_res_dir, dir_name, f"{strategy}.csv")
                hit1 = load_result(fp)
                auc = metrics.auc(anno_percent_list, hit1)
                auc_list1.append(auc)
            ax.plot(alpha_range, auc_list1, label=f"{model_name}%", color=colors[m_idx], marker=marker_list[m_idx])
            if percent == 0.0:
                ax.set_title(f"struct_uncert on DW({int(100*percent)}%)", fontsize=23)
            else:
                ax.set_title(f"ActiveEA on DW({int(100 * percent)}%)", fontsize=23)
            ax.set_xlabel(r'$ \alpha $', fontsize=20)

            ax.tick_params(axis="x", labelsize=15)
            ax.tick_params(axis="y", labelsize=15)
    axes[0,0].set_ylim(15, 30)
    axes[0,1].set_ylim(15, 30)
    axes[0,0].set_ylabel("AUC@0.5", fontsize=20)

    task_group = "effect_of_batchsize" # overall_perf, effect_of_alpha, effect_of_batchsize
    task_res_dir = os.path.join(res_dir, task_group)
    width = 0.4
    for m_idx in range(len(ea_model_list)):
        model_name = ea_model_list[m_idx]

        for p_idx in range(len(bach_percent_list)):
            percent = bach_percent_list[p_idx]
            ax = axes[1, p_idx]

            if percent == 0.0:
                auc_list1 = []
                for batchsize in batchsize_range:
                    strategy = f"STRUCT_UNCER_alpha{0.1}_batchsize{batchsize}"
                    dir_name = f"{data_name}_BACH{percent}_{model_name}"
                    fp = os.path.join(task_res_dir, dir_name, f"{strategy}.csv")
                    hit1 = load_result(fp)
                    auc = metrics.auc(anno_percent_list, hit1)
                    auc_list1.append(auc)
                ax.plot(batchsize_range, auc_list1, label=f"{model_name}%", color=colors[m_idx], marker=marker_list[m_idx])
                # ax.bar(np.arange(len(auc_list1))-width/2, auc_list1, width=width, label=f"{int(100*percent)}%", color=colors[p_idx])
            elif percent > 0:
                auc_list1 = []
                for batchsize in batchsize_range:
                    strategy = f"STRUCT_UNCER_BACH_RECOG_CV_alpha{0.1}_batchsize{batchsize}"
                    dir_name = f"{data_name}_BACH{percent}_{model_name}"
                    fp = os.path.join(task_res_dir, dir_name, f"{strategy}.csv")
                    hit1 = load_result(fp)
                    auc = metrics.auc(anno_percent_list, hit1)
                    auc_list1.append(auc)
                ax.plot(batchsize_range, auc_list1, label=f"{model_name}%", color=colors[m_idx], marker=marker_list[m_idx])
                # ax.bar(np.arange(len(auc_list1))+width/2, auc_list1, width=width, label=f"{int(100 * percent)}%", color=colors[p_idx])
                # ax.set_xticks(range(len(auc_list1)))
                # ax.set_xticklabels([str(b) for b in batchsize_range])

            # ax.set_title(f"{model_name} on DW", fontsize=25)

            ax.set_xlabel("batch size of sampling", fontsize=20)
            ax.tick_params(axis="x", labelsize=15)
            ax.tick_params(axis="y", labelsize=15)


    axes[1,0].set_ylim(20, 30)
    axes[1,1].set_ylim(20, 30)
    axes[1,0].set_ylabel("AUC@0.5", fontsize=20)


    # lines, labels = axes[1,1].get_legend_handles_labels()
    # fig.legend(lines, labels, loc="lower center", ncol=2, fontsize=20, bbox_to_anchor=(0.5, -0.1))
    axes[0, 1].legend(loc="upper right", ncol=2, fontsize=15)

    fig.tight_layout()
    plt.savefig(os.path.join(fig_save_dir, "sensitivity_of_param.pdf"), pad_inches=0, bbox_inches="tight")
    fig.show()




def draw_effect_of_bachelor(res_dir):
    data_name = "D_W_15K_V1"
    # bach_percent_list = ["0.0", "0.3"]
    bach_percent_list = [0.0, 0.1, 0.2, 0.3, 0.4]
    ea_model_list = ["BootEA", "Alinet"]
    strategy_name_list = ["RAND", "DEGREE", "PAGERANK", "BETWEEN", "UNCERTAINTY", "STRUCT_UNCER_alpha0.1_batchsize100",
                          "STRUCT_UNCER_BACH_RECOG_CV_alpha0.1_batchsize100"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    marker_list = ["*", "^", "o", ">", "s", "D", "v"]
    colors = sbn.color_palette()

    anno_percent_list = [0.00, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
    ylims = [[(0, 100), (0, 100)], [(0, 100), (0, 100)], [(30, 80), (65, 95)]]

    strategy_name_in_fig_list = ["rand", "degree", "pagerank", "betweenness", "uncertainty", "struct_uncert",
                                 "ActiveEA"]
    data_name_in_fig_list = ["DW", "ENDE"]

    for m_idx in range(len(ea_model_list)):
        ea_model = ea_model_list[m_idx]
        ax = axes[m_idx]


        for s_idx in range(len(strategy_name_list)):
            strategy = strategy_name_list[s_idx]
            auc_list = []
            x_list = []
            for p_idx in range(len(bach_percent_list)):
                percent = bach_percent_list[p_idx]

                dir_name = f"{data_name}_BACH{percent}_{ea_model}"
                fp = os.path.join(res_dir, dir_name, f"{strategy}.csv")
                if not os.path.exists(fp):
                    print("dont have ", fp)
                    continue
                x_list.append(percent)
                hit1 = load_result(fp)
                auc = metrics.auc(anno_percent_list, hit1)
                auc_list.append(auc)

            ax.plot(x_list, auc_list, label=strategy_name_in_fig_list[s_idx], color=colors[s_idx],
                        marker=marker_list[s_idx])
            ax.set_title(f"{ea_model} on DW", fontsize=25)
            ax.grid()
            ax.tick_params(axis="x", labelsize=15)
            ax.tick_params(axis="y", labelsize=15)
            ax.set_xlabel("Proportion of Bachelors", fontsize=20)
            ax.set_ylabel("AUC@0.5", fontsize=20)


    axes[0].set_ylim(10, 30)
    axes[1].set_ylim(10, 30)


    lines, labels = axes[1].get_legend_handles_labels()
    fig.legend(lines, labels, loc="lower left", ncol=4, fontsize=15, bbox_to_anchor=(0.1, -0.15))
    fig.tight_layout()
    plt.savefig(os.path.join(fig_save_dir, "effect_of_bachelor.pdf"), pad_inches=0.05, bbox_inches="tight")
    fig.show()



def effectiveness_of_bachelor_recog(res_dir):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    def get_micro_f1_wrt_ite_no(fn):
        with open(fn) as file:
            lines = file.read().strip().split("\n")
            f1_map = {
                "cv_0": [],
                "cv_1": [],
                "cv_2": [],
                "cv_3": [],
                "cv_4": [],
                "ave": []
            }
            for line in lines:
                obj = json.loads(line)
                for k in f1_map.keys():
                    valid_metrics_list = obj[k]["valid_metrics"]
                    test_metrics_list = obj[k]["test_metrics"]
                    valid_micro_f1_list = [vm[3][2] for vm in valid_metrics_list]
                    test_micro_f1_list = [vm[3][2] for vm in test_metrics_list]
                    max_idx = np.argmax(valid_micro_f1_list)
                    validf1 = valid_micro_f1_list[max_idx]
                    test_f1 = test_micro_f1_list[max_idx]
                    # valid_f1, test_f1 = get_micro_f1(k)
                    f1_map[k].append(test_f1)
            return f1_map

    model_name_list = ["BootEA", "Alinet"]
    for m_idx in range(len(model_name_list)):
        model_name = model_name_list[m_idx]
        variant = f"D_W_15K_V1_BACH0.3_{model_name}"
        res_fn = os.path.join(res_dir, variant, "bach_recog_cv/bach_perf_detail_logs.json")
        f1_map = get_micro_f1_wrt_ite_no(res_fn)

        step_list = list(range(len(f1_map["ave"])))
        sample_ite_no_list = [step*5+1 for step in step_list]
        ax = axes[m_idx]
        colors = sbn.color_palette()
        keys = list(f1_map.keys())
        for idx in range(len(keys)):
            k = keys[idx]
            if k == "ave":
                ax.plot(sample_ite_no_list, f1_map[k], color="green", label="with ME")
            else:
                ax.scatter(sample_ite_no_list, f1_map[k], color="red", alpha=0.3, s=10, label="w/o ME")
        ax.set_ylim(0.3, 1)
        ax.set_title(f"{model_name} on DW (30%)", fontsize=25)
        ax.tick_params(axis="x", labelsize=15)
        ax.tick_params(axis="y", labelsize=15)
        ax.set_xlabel("Sampling iteration", fontsize=20)
        ax.set_ylabel("Micro-F1", fontsize=20)
        min_ite = sample_ite_no_list[0]
        max_ite = sample_ite_no_list[-1]
        ax.set_xticks( np.arange(min_ite, max_ite+1, 10))

    lines, labels = axes[1].get_legend_handles_labels()
    axes[0].legend(lines[:2], labels[:2], fontsize=15)
    fig.tight_layout()
    plt.savefig(os.path.join(fig_save_dir, "effectiveness_of_bachelor_recog.pdf"), pad_inches=0.05, bbox_inches="tight")
    fig.show()








proj_dir = ""
output_dir = os.path.join(proj_dir, "output")
fig_save_dir = os.path.join(proj_dir, "output/old/results/figures")
res_dir = os.path.join(proj_dir, "output/old/results/")

task_group = "overall_perf" # overall_perf, effect_of_alpha
task_res_dir = os.path.join(res_dir, task_group)
# draw_overall_performance(task_res_dir)
# draw_overall_performance_on_enfr100k(task_res_dir)
# draw_overall_performance_2x6(task_res_dir)
# calculte_overall_performance_auc(task_res_dir)


print_overall_performance_auc_table(output_dir)


# task_group = "effect_of_bayesian" # overall_perf, effect_of_alpha
# overall_task_res_dir = task_res_dir
# task_res_dir = os.path.join(res_dir, task_group)
# draw_effect_of_bayesian2(task_res_dir, overall_task_res_dir)



# task_group = "effect_of_alpha" # overall_perf, effect_of_alpha
# task_res_dir = os.path.join(res_dir, task_group)
# draw_effect_of_alpha(task_res_dir)
#
#
# task_group = "effect_of_batchsize" # overall_perf, effect_of_alpha, effect_of_batchsize
# res_dir = os.path.join(proj_dir, "output/results/", task_group)
# draw_effect_of_batchsize()

# draw_sensitivity_of_parameters(res_dir)


# task_group = "effect_of_bachpercent" # overall_perf, effect_of_alpha, effect_of_batchsize
# task_res_dir = os.path.join(res_dir, task_group)
# draw_effect_of_bachelor(task_res_dir)
#
#
# task_group = "perf_of_bachelor_recog" # overall_perf, effect_of_alpha, effect_of_batchsize
# task_res_dir = os.path.join(res_dir, task_group)
# effectiveness_of_bachelor_recog(task_res_dir)

