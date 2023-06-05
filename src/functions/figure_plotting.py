# -*- coding: utf-8 -*-
"""
Plotting of population data
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.metrics import confusion_matrix
import shap
import yaml


class FigurePlot:
    """FigurePlot

    Plotting population data

    Parameters to constructor:
    configuration_file ... a yaml file with configuration parameters

    Public functions:
    plot_two ........ plot original and synthetic in same frame horisontal
    plot_two_flat ... plot original and synthetic in same frame standing
    plot_compare .... plot regressionline between two datasets variables
    plot_dist ....... plot sns.displot for a single variable
    plot_catplot .... plot sns.catplot for countable or scaled data

    """
    def __init__(self,
                 configuration_file
                 ):
        self.configuration_file = configuration_file
        self.cfg = None
        self.fig_dpi = None
        self.CB91_Green = None
        self.CB91_Light_Green = None
        self.CB91_Brown_Gray = None
        self.CB91_Red = None
        self.CB91_Brown = None
        self.CB91_Purple = None
        self.CB91_Violet = None
        self.CB91_Yellow = None
        self.initiate()

    def initiate(self):       
        self.set_configurations()
    
    def set_configurations(self):
        """Read and set values from yaml configuration file

        Parameters
        ----------

        Returns
        -------
        None

        """
        with open(self.configuration_file, 'r') as file:
            self.cfg = yaml.safe_load(file)
        # read configurations
        palette = self.cfg["palette"]
        names = self.cfg["color_list_names"]
        assert len(palette) == len(names)
        self.CB91_Green = palette[names[0]]
        self.CB91_Light_Green = palette[names[1]]
        self.CB91_Brown_Gray = palette[names[2]]
        self.CB91_Red = palette[names[3]]
        self.CB91_Brown = palette[names[4]]
        self.CB91_Purple = palette[names[5]]
        self.CB91_Violet = palette[names[6]]
        self.CB91_Yellow = palette[names[7]]
        self.fig_dpi = self.cfg["fig_dpi"]
        
    def shap_summary(self, shap_values, features,
                     title, model_type, model_name, model, 
                     save=False):
        shap.summary_plot(shap_values=shap_values,
                          features=features,
                          feature_names=features.columns,
                          cmap="RdGy_r",
                          plot_size=[10, 8],
                          show=False)
        plt.gcf().axes[-1].set_aspect(100)
        plt.gcf().axes[-1].set_box_aspect(100)
        if save:
            plt.savefig('figures/{}/shap_summary_{}_{}_{}_{}.png'.format(model,
                                                                         title,
                                                                         model_type,
                                                                         model,
                                                                         model_name
                                                                         ), dpi=self.fig_dpi)
        plt.tight_layout()
        plt.show()
        
    def bland_altman(self, df_original, df_synthetic,
                     title, model_type, model_name, model, 
                     save=False, sd_limit=1.96):
        """Plot Bland-Altman for the differences between original and synthetic data.
        Uses statsmodel to compute and display Bland-Altman.

        Parameters
        ----------
        df_original .... original data (Pandas dataframe)
        df_synthetic ... synthetic data
        title .......... title
        model_type ..... model_type for synthetic data
        model_name ..... model_name for synthetic data
        model .......... model for synthetic data
        save ........... if model should be saved

        Returns
        -------
        Visuals and saved file if save=True
        """

        f, ax = plt.subplots(1, figsize=(8, 6), dpi=self.fig_dpi)
        sm.graphics.mean_diff_plot(df_synthetic.mean(), df_original.mean(), ax=ax, sd_limit=sd_limit)
        plt.gcf.set_color = self.CB91_Brown_Gray
        if save:
            plt.savefig('figures/{}/bland_altman_{}_{}_{}_{}.png'.format(model,
                                                                         title,
                                                                         model_type,
                                                                         model,
                                                                         model_name
                                                                         ), dpi=self.fig_dpi)
        plt.tight_layout()
        plt.show()
        
    # Source: https://www.appsloveworld.com/machine-learning/97/modified-blandaltman-plot-in-seaborn
    def color_bland_altman(self, df_original, df_synthetic,
                           title, model_type, model_name, model,
                           difference_from=None,
                           sd_limit=1.96,
                           upper_limit=0.13,
                           lower_limit=-0.13,
                           save=False):
        if difference_from is None:
            difference_from = "original"
        predicted = np.asarray(df_synthetic.mean())
        truth = np.asarray(df_original.mean()) 
        diff = predicted - truth
        fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
        sm.graphics.mean_diff_plot(df_synthetic.mean(),
                                   df_original.mean(),
                                   ax=ax,
                                   sd_limit=sd_limit,
                                   scatter_kwds={"s": 1,
                                                 "alpha": 0.1})
        ax.scatter(truth, diff, s=30, c=truth, cmap="RdGy_r")
        plt.ylim(upper_limit, lower_limit)
        ax.set_xlabel(difference_from)
        ax.set_ylabel('difference from ' + difference_from)
        ax.set_title(title)
        if save:
            plt.savefig('figures/{}/color_bland_altman_{}_{}_{}_{}.png'.format(model,
                                                                               title,
                                                                               model_type,
                                                                               model,
                                                                               model_name
                                                                               ), dpi=self.fig_dpi)
        plt.tight_layout()
        plt.show()

    def plot_two(self, df_original, df_synthetic,
                 title, hue_value, n_bins,
                 model_type, model, model_name, save=False,
                 scale_max=11000, scale_min=0):
        """Plot original and synthetic in same frame on a common variable
        "title" and a selected binary variable "hue_value". Calculate
        SRMSE, Pearson's and R2 on the correlation between variables,
        and display this in a textbox.

        Parameters
        ----------
        df_original ....... original data
        df_synthetic ...... synthetic data
        title ............. the name of the common variable to display
        hue_value ......... a binary variable to split the main variable
        n_bins ............ the number of categories of the variable
        model_type ........ custom like "mixed" "categorical"
        model ............. i.e vae or gan
        model_name ........ i.e the config of layers like 100-50-25
        save .............. True or False (to save file locally)

        Returns
        -------
        A plot to display and storage of the plot as png in a folder
        figure/<model_type>_<title>_<model>_<model_name>.png

        """
        fig, axs = plt.subplots(1, 2, 
                                figsize=(6, 3),
                                sharey="all",  # valid "all" "row" "col" "none"
                                gridspec_kw=dict(width_ratios=[3, 3]),
                                dpi=self.fig_dpi)
        sns.set_style("whitegrid", {'axes.grid': False})
        # customised hue colors  
        palette_original = {0: self.CB91_Light_Green,
                            1: self.CB91_Green
                            }
        palette_synthetic = {0: self.CB91_Brown_Gray,
                             1: self.CB91_Red
                             }
        # Original data (test) x-axis
        g1 = sns.histplot(data=df_original,
                          bins=n_bins,
                          hue=hue_value,
                          multiple="dodge",
                          y=title,
                          # kde=True,
                          # legend=False,
                          palette=palette_original,
                          ax=axs[0],
                          )
        count1, bin1 = np.histogram(df_original[title],
                                    bins=n_bins)
        count2, bin2 = np.histogram(df_synthetic[title],
                                    bins=bin1)
        # Synthetic data (y = predict) y-axis
        g2 = sns.histplot(data=df_synthetic,
                          bins=bin1,
                          # color="seagreen",
                          hue=hue_value,
                          y=title,
                          multiple="dodge",
                          # kde=True,
                          palette=palette_synthetic,
                          ax=axs[1]
                          )
        g1.set(xlabel=None)  # remove the axis label
        g2.set(xlabel=None)  # remove the axis label
        sns.move_legend(g1, "upper right")
        sns.move_legend(g2, "upper right")
        g1.set(xlim=(scale_min, scale_max))
        g2.set(xlim=(scale_min, scale_max))
        axs[0].yaxis.get_major_locator().set_params(integer=True)
               
        # COMPUTE ERRORS

        y_mean = count1.mean()
        corr = np.corrcoef(count1, count2)[0, 1]
        rmse = ((count1 - count2) ** 2).mean() ** .5
        srmse = rmse/y_mean  # y_test.mean()
        # r-square
        reg = sm.OLS(count2, count1)
        res = reg.fit()
        r2_ = res.rsquared
        # PREPARE TEXT
        text = ["SRMSE",
                "{:.3f}".format(srmse),
                "Pearson's",
                "{:.3f}".format(corr),
                "R2",
                "{:.3f}".format(r2_)
                ]
        text = '\n'.join(text)
        left, width = .25, .4
        bottom, height = .1, .4
        right = left + width
        # top = bottom + height
        g2.text(right,
                bottom,
                text,
                # rotation=-30,
                wrap=True,
                fontsize=10,
                bbox=dict(facecolor='white', 
                          boxstyle='round, pad=0.5, rounding_size=0.2',
                          edgecolor=self.CB91_Green,
                          alpha=0.5,
                          snap=False),
                # size='small',
                # horizontalalignment="right",
                # ha="right",
                horizontalalignment='left',
                verticalalignment='bottom',
                transform=axs[1].transAxes,
                color="gray", 
                # weight="semibold"
                )
        if save:
            plt.savefig('figures/{}/marginals_{}_{}_{}_{}.png'.format(model,
                                                                      title,
                                                                      model_type,
                                                                      model,
                                                                      model_name
                                                                      ), dpi=self.fig_dpi)
        plt.tight_layout()
        plt.show()
        
    def plot_two_flat(self, df_original, df_synthetic,
                      title, hue_value, n_bins, model_type,
                      model, model_name, save=False, 
                      scale_max=10000, kde=True):
        """Plot original and synthetic in same frame on a common variable
        "title" and a selected binary variable "hue_value". Calculate
        SRMSE, Pearson's and R2 on the correlation between variables,
        and display this in a textbox.

        Parameters
        ----------
        df_original ....... original data
        df_synthetic ...... synthetic data
        title ............. common variable to display
        hue_value ......... a binary variable to split the main variable
        n_bins ............ the number of categories of the variable
        model_type ........ custom like "mixed" "categorical"
        model ............. i.e vae or gan
        model_name ........ i.e the config of layers like 100-50-25
        save .............. True or False (to save file locally)

        Returns
        -------
        A plot to display and storage of the plot as png in a folder
        figure/<model_type>_<title>_<model>_<model_name>.png

        """
        fig, axs = plt.subplots(1, 2, 
                                figsize=(6, 3),
                                sharey="all",  # valid "all" "row" "col" "none"
                                gridspec_kw=dict(width_ratios=[3, 3]),
                                dpi=self.fig_dpi)
        sns.despine()
        sns.set_style("whitegrid", {'axes.grid': False})
        # customised hue colors  
        palette_original = {0: self.CB91_Light_Green,
                            1: self.CB91_Green
                            }
        palette_synthetic = {0: self.CB91_Brown_Gray,
                             1: self.CB91_Red
                             }
        # Original data (test) x-axis
        g1 = sns.histplot(data=df_original,
                          bins=n_bins,
                          hue=hue_value,
                          multiple="dodge",
                          x=title,
                          kde=kde,
                          # egend=False,
                          palette=palette_original,
                          ax=axs[0],
                          )
        count1, bin1 = np.histogram(df_original[title],
                                    bins=n_bins)
        count2, bin2 = np.histogram(df_synthetic[title],
                                    bins=bin1)
        # Synthetic data (y = predict) y-axis
        g2 = sns.histplot(data=df_synthetic,
                          bins=bin1,
                          # color="seagreen",
                          hue=hue_value,
                          x=title,
                          multiple="dodge",
                          kde=kde,
                          palette=palette_synthetic,
                          ax=axs[1]
                          )
        g1.set(xlabel=None)  # remove the axis label
        g2.set(xlabel=None)  # remove the axis label
        sns.move_legend(g1, "upper right")
        sns.move_legend(g2, "upper right")
        g1.set(xlim=(0, scale_max))
        g2.set(xlim=(0, scale_max))
        axs[0].yaxis.get_major_locator().set_params(integer=True)
        # COMPUTE ERRORS
        y_mean = count1.mean()
        corr = np.corrcoef(count1, count2)[0, 1]
        rmse = ((count1 - count2) ** 2).mean() ** .5
        srmse = rmse/y_mean  # y_test.mean()
        # r-square
        reg = sm.OLS(count2, count1)
        res = reg.fit()
        r2_ = res.rsquared
        # PREPARE TEXT
        text = ["SRMSE",
                "{:.3f}".format(srmse),
                "Pearson's",
                "{:.3f}".format(corr),
                "R2",
                "{:.3f}".format(r2_),
                "n_bins"
                ]
        text = '\n'.join(text)
        g2.text(-0.8, 
                -0.1, 
                text,
                # rotation=-30,
                wrap=True,
                fontsize=10,
                bbox=dict(facecolor='white', 
                          boxstyle='round, pad=0.5, rounding_size=0.2',
                          edgecolor=self.CB91_Green,
                          alpha=0.5,
                          snap=False),
                # size='small',
                # horizontalalignment="right",
                # ha="right",
                color="gray", 
                # weight="semibold"
                )
        if save:
            plt.savefig('figures/{}/marginals_{}_{}_{}_{}.png'.format(model,
                                                                      title,
                                                                      model_type,
                                                                      model,
                                                                      model_name
                                                                      ), dpi=self.fig_dpi)
        plt.show()
        
    def plot_compare(self, data, title, model_type,
                     model, model_name, scale_min=None,
                     scale_max=None, save=False, norm_rmse=True):
        """Plot original and synthetic regression-line. Calculate
        SRMSE, Pearson's and R2 on the correlation between variables,
        and display this in a textbox.

        Parameters
        ----------
        df_original ....... original data
        df_synthetic ...... synthetic data
        title ............. any name to plot file
        model_type ........ custom like "mixed" "categorical"
        model ............. i.e vae or gan
        model_name ........ i.e the config of layers like 100-50-25
        save .............. True or False (to save file locally)

        Returns
        -------
        A plot to display and storage of the plot as png in a folder
        figure/<title>_<model>_<model_name>.png

        """
        columns = list(data.columns)  # columns[0] is "Original"
        g = sns.lmplot(data=data, x=columns[0], y=columns[1],
                       height=8,
                       fit_reg=True,
                       line_kws={'color': self.CB91_Green},
                       facet_kws=dict(sharex=False, sharey=False)
                       )
        sns.despine()
        sns.set_style("whitegrid", {'axes.grid': False})
        count1 = data[columns[0]]
        count2 = data[columns[1]]
        y_mean = count1.mean()
        corr = np.corrcoef(count1, count2)[0, 1]
        rmse = ((count1 - count2) ** 2).mean() ** .5
        if norm_rmse:
            srmse = rmse/abs(y_mean)  # y_test.mean()
        # r-square
        reg = sm.OLS(count2, count1)
        res = reg.fit()
        r2_ = res.rsquared
        # PREPARE TEXT
        if norm_rmse:
            text = ["SRMSE",
                    "{:.3f}".format(srmse),
                    "Pearson's",
                    "{:.3f}".format(corr),
                    "R2",
                    "{:.3f}".format(r2_)
                    ]
        else:
            text = ["RMSE",
                    "{:.3f}".format(rmse),
                    "Pearson's",
                    "{:.3f}".format(corr),
                    "R2",
                    "{:.3f}".format(r2_)
                    ]
        text = '\n'.join(text)
        left, width = .25, .4
        bottom, height = .1, .4
        right = left + width
        # top = bottom + height
        g.axes[0, 0].text(right,
                          bottom,
                          text,
                          wrap=True,
                          fontsize=18,
                          bbox=dict(facecolor='white',
                                    boxstyle='round, pad=0.5, rounding_size=0.2',
                                    edgecolor=self.CB91_Green,
                                    alpha=0.5,
                                    snap=False),
                          horizontalalignment='left',
                          verticalalignment='bottom',
                          # transform=axs[1].transAxes,
                          color="gray",
                          )
        if scale_min is None:
            scale_min = 0
        if scale_max is None:
            scale_max = 1
        g.set(ylim=(int(scale_min), int(scale_max)))
        g.set(xlim=(int(scale_min), int(scale_max)))
        # g.set(yticks=list(range(int(scale_min), int(scale_max))))
        if save:
            plt.savefig('figures/{}/compare_{}_{}_{}_{}.png'.format(model,
                                                                    title,
                                                                    model_type,
                                                                    model,
                                                                    model_name
                                                                    ), dpi=self.fig_dpi)
        plt.tight_layout()
        plt.show()

    def plot_dist(self, data, title, model_type, model, model_name,
                  scale_min=None, scale_max=None,
                  bins=None, save=False, kde=True):
        """Plot one variable by seaborn displot

        Parameters
        ----------
        data ....... data containing variable "title"
        title ...... name of variable
        model_type ........ custom like "mixed" "categorical"
        model ............. i.e vae or gan
        model_name ........ i.e the config of layers like 100-50-25
        save .............. True or False (to save file locally)
        kde ............... plot kde is default True

        Returns
        -------
        A plot to display and storage of the plot as png in a folder
        figure/dist_<title>_<data_type>_<model>_<model_name>.png

        """
        sns.despine()
        sns.set_style("whitegrid", {'axes.grid': False})
        g = sns.displot(data=data,
                        y=title,
                        # bins=bins,
                        discrete=True,
                        rug=True,
                        height=4,
                        kde=kde,
                        )
        if scale_min is None:
            scale_min = 0
        if scale_max is None:
            scale_max = int(bins)
        g.set(ylim=(int(scale_min), int(scale_max)))
        g.set(yticks=list(range(int(scale_min), int(scale_max))))
        if save:            
            plt.savefig('figures/{}/dist_{}_{}_{}_{}.png'.format(model,
                                                                 title,
                                                                 model_type,
                                                                 model,
                                                                 model_name
                                                                 ), dpi=self.fig_dpi)
        plt.tight_layout()
        plt.show()

    def plot_catplot(self, data, title, model_type, model, model_name,
                     save=False, kind="box"):
        """Plot a seaborn catplot. Input data have to be a pandas dataframe
        with column names that will be displayed. Data must be
        countable or scaled.

        Parameters
        ----------
        data .............. data to be displayed
        model_type ........ custom like "mixed" "categorical"
        model ............. i.e vae or gan
        model_name ........ i.e the config of layers like 100-50-25
        save .............. True or False (to save file locally)

        Returns
        -------
        A plot to display and storage of the plot as png in a folder
        figure/<model_type>_<title>_<model>_<model_name>.png

        """
        sns.despine()
        sns.set_style("whitegrid", {'axes.grid': False})
        g = sns.catplot(data=data,
                        y=None,
                        kind=kind,
                        height=12)
        g.set_xticklabels(labels=data.columns, rotation=40)
        if save:
            plt.savefig('figures/{}/catplot_{}_{}_{}_{}.png'.format(model,
                                                                    title,
                                                                    model_type,
                                                                    model,
                                                                    model_name
                                                                    ), dpi=self.fig_dpi)
        plt.tight_layout()
        plt.show()

    def jointplot(self,
                  df,
                  title_x,
                  title_y,
                  model_type,
                  model,
                  model_name,
                  scale_min=None,
                  scale_max=None,
                  save=False
                  ):
        sns.despine()
        if scale_min is None:
            scale_min = 0
        if scale_max is None:
            scale_max = 1
        sns.set_style("whitegrid", {'axes.grid': False})

        g = sns.jointplot(data=df,
                          x=title_x,
                          y=title_y,
                          # multiple="dodge",
                          )
        g.ax_marg_x.set_xlim(scale_min, scale_max)
        g.ax_marg_y.set_ylim(scale_min, scale_max)

        if save:
            plt.savefig('figures/{}/joint_{}_{}_{}_{}.png'.format(model,
                                                                  title_x,
                                                                  model_type,
                                                                  model,
                                                                  model_name
                                                                  ), dpi=self.fig_dpi)
        plt.tight_layout()
        plt.show()

    def contingency_plot(self, dta,
                         columns,
                         labels,
                         title,
                         model_type,
                         model,
                         model_name,
                         confusion_matrix=False,
                         annotate=False,
                         save=False):
        cross = pd.crosstab(dta[columns[0]], dta[columns[1]], margins=True)
        chi_ = stats.chi2_contingency(cross)
        for c in chi_[3]:
            c = np.array(c)
        chi = np.array(chi_[3])
        ax = plt.subplot()
        sns.heatmap(chi/dta.shape[0], annot=annotate, fmt='.2f', ax=ax, cmap="Greens")
        ax.set_ylabel(columns[0])
        ax.set_xlabel(columns[1])
        if confusion_matrix:
            heading = "Confusion Matrix "
        else:
            heading = "Contingency Table "
        ax.set_title(heading + columns[0] + " + " + columns[1] + " " + title)
        if labels is not None:
            ax.xaxis.set_ticklabels(labels[1])
            ax.yaxis.set_ticklabels(labels[0])
        if save:
            plt.savefig('figures/{}/contingency_{}_{}_{}_{}_{}_{}.png'.format(model,
                                                                              columns[0],
                                                                              columns[1],
                                                                              title,
                                                                              model_type,
                                                                              model,
                                                                              model_name
                                                                              ), dpi=self.fig_dpi)
        plt.tight_layout()
        plt.show()

    def confusion_matrix_plot(self, df_o_col, df_s_col,
                              title,
                              model_type,
                              model,
                              model_name,
                              latent,
                              annotate=False,
                              normalise=False,
                              decimals=".0f",  # options "g", ":.0.xxx"f
                              save=False
                              ):
        values = len(df_o_col.value_counts())
        labels = np.arange(1, values + 1, 1)
        confusion = confusion_matrix(df_o_col.values, df_s_col.values)
        ax = plt.subplot()
        if normalise:
            # Sum up to one for each row = original variable (showing proportions assigned by generative model)
            confusion = confusion.astype('float') / confusion.sum(axis=1)[:, np.newaxis]
        sns.heatmap(confusion, annot=annotate, fmt=decimals, ax=ax, cmap="Greens")
        ax.set_xlabel = "Original"
        ax.set_ylabel = model + "-" + str(latent)
        ax.set_title("Confusion Matrix " + str(np.squeeze(df_o_col.columns)) +
                     " " + title.upper() + "-" + str(latent))
        ax.xaxis.set_ticklabels(labels)
        ax.yaxis.set_ticklabels(labels)
        if save:
            plt.savefig('figures/{}/confusion_{}_{}_{}_{}_{}.png'.format(model,
                                                                         str(np.squeeze(df_o_col.columns)),
                                                                         title,
                                                                         model_type,
                                                                         model,
                                                                         model_name
                                                                         ), dpi=self.fig_dpi)
        plt.tight_layout()
        plt.show()
