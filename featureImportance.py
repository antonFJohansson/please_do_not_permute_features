import matplotlib.pyplot as plt
import random
import numpy as np


class feature_importance():
    
  """
  Class to obtain feature importance and variable importance with the help of
      a Gaussian Process based regression method. Can probably easily be
      extended to other functions that return variance estimate in addition
      to a prediction.
  Args:
      gp: A trained Gaussian process model
      X: The array of features
      y: The corresponding target array
      feature_names: The names of each feature
  """

  def __init__(self, gp, X, y, feature_names = None):
    self.gp = gp
    self.X = X
    self.y = y
    self.feature_names = feature_names
    if self.feature_names is None:
      self.feature_names = ['Feature {}'.format(i) for i in range(X.shape[1])]

  def get_mean_and_std(self, p,c):
      
    ## Obtain the mean and std for Partial Dependency plot
        ## Need to take covariance into account
    
    mean_pred = np.mean(p)
    c_zero_diag = np.copy(c)
    diag = np.diag(c)
    c_zero_diag = c_zero_diag - np.diag(diag)
    std = np.sqrt(1/(diag.shape[0]**2)*(np.sum(c_zero_diag) + np.sum(diag)))

    max_pred = mean_pred + 1.96*std
    min_pred = mean_pred - 1.96*std
    return mean_pred, max_pred, min_pred

  def get_PD_plot(self, feature_ind, fig_size_x = 9, fig_size_y = 6,
                  x_max_lim = None, x_min_lim = None):
    
    """
        Function to obtain a Partial Dependency plot.
        Args:
            feature_ind: list with feature indices that are to be plotted
            fig_size_x: Size in x-width for one plot
            fig_size_y: Size in y-width for one plot
            x_max_lim: The maximal x-value to be plotted to
            x_min_lim: The minimal x-value to be plotted to
    """
    
    figure, ax = plt.subplots(1,len(feature_ind), figsize= (len(feature_ind)*fig_size_x, fig_size_y))

    for idx,feat_id in enumerate(feature_ind):

      x_min = np.min(self.X[:, feat_id]) if x_min_lim is None else x_min_lim
      x_max = np.max(self.X[:, feat_id]) if x_max_lim is None else x_max_lim
      x_pi = np.linspace(x_min, x_max, 100)

      mean_pred_list = []
      max_pred_list = []
      min_pred_list = []
      for iii in range(x_pi.shape[0]):
        new_X_PI = self.X.copy()
        new_X_PI[:, feat_id] = x_pi[iii]
        p, c = self.gp.predict(new_X_PI, return_cov=True)
        mean_p, max_p, min_p = self.get_mean_and_std(p,c)

        mean_pred_list.append(mean_p)
        max_pred_list.append(max_p)
        min_pred_list.append(min_p)

      if len(feature_ind) > 1:
          ax[idx].plot(x_pi, mean_pred_list)
          ax[idx].fill_between(x_pi, min_pred_list, max_pred_list, alpha = 0.2)
          ax[idx].set_ylabel('Partial Dependence')
          ax[idx].set_xlabel(self.feature_names[feat_id])
          ax[idx].grid(axis='y')
      else:
          ax.plot(x_pi, mean_pred_list)
          ax.fill_between(x_pi, min_pred_list, max_pred_list, alpha = 0.2)
          ax.set_ylabel('Partial Dependence')
          ax.set_xlabel(self.feature_names[feat_id])
          ax.grid(axis='y')

  def get_ICE_plot(self, feature_ind, num_samples, fig_size_x = 9, fig_size_y = 6,
                   x_max_lim = None, x_min_lim = None):
      
    """
        Function to obtain an ICE plot.
        Args:
            feature_ind: list with feature indices that are to be plotted
            fig_size_x: Size in x-width for one plot
            fig_size_y: Size in y-width for one plot
            x_max_lim: The maximal x-value to be plotted to
            x_min_lim: The minimal x-value to be plotted to
    """

    ## Obtain the random rows to display
    row_inds_list = [iii for iii in range(self.X.shape[0])]
    random.shuffle(row_inds_list)
    row_inds_list = row_inds_list[0:num_samples]
    
    figure, ax = plt.subplots(1,len(feature_ind), figsize= (len(feature_ind)*fig_size_x, fig_size_y))

    X_PI = self.X.copy()
    for idx, feat_id in enumerate(feature_ind):
    
    
      full_list_pred = []
      for row_ind in row_inds_list:
      
        
        x_curr = X_PI[row_ind, :]
        x_curr = x_curr.reshape(1,-1)
        
        x_min = np.min(X_PI[:, feat_id]) if x_min_lim is None else x_min_lim
        x_max = np.max(X_PI[:, feat_id]) if x_max_lim is None else x_max_lim
        
        x_pi = np.linspace(x_min, x_max, 100)
        curr_pred_arr = np.zeros((3, x_pi.shape[0]))

        for jjj in range(x_pi.shape[0]):
          x_curr_ice = x_curr.copy()
          x_curr_ice[0,feat_id] = x_pi[jjj]
          p, c = self.gp.predict(x_curr_ice, return_cov=True)
          curr_pred_arr[0,jjj] = p
          curr_pred_arr[1,jjj] = p + np.sqrt(c)*1.96
          curr_pred_arr[2,jjj] = p - np.sqrt(c)*1.96
        full_list_pred.append(curr_pred_arr)
        
      if len(feature_ind) > 1:
          for arr in full_list_pred:
            ax[idx].plot(x_pi, arr[0,:])
            ax[idx].fill_between(x_pi, arr[1,:], arr[2,:], alpha = 0.2)
          ax[idx].set_ylabel('ICE')
          ax[idx].set_xlabel(self.feature_names[feat_id])
          ax[idx].grid(axis='y')
      else:
          for arr in full_list_pred:
            ax.plot(x_pi, arr[0,:])
            ax.fill_between(x_pi, arr[1,:], arr[2,:], alpha = 0.2)
          ax.set_ylabel('ICE')
          ax.set_xlabel(self.feature_names[feat_id])
          ax.grid(axis='y')
           

  def loss_func(self, fx, y, single_sample = False):

    ## MSE-loss. Can easily be extended to something else

    if single_sample:
      return (fx - y)**2
    else:
      return np.sum((fx - y)**2)

  def get_VI_plot(self, feature_ind = None, num_samples = 100):

    """
    Function to get variable importance plot.
    Args: feature_ind: list with feature indices that are to be plotted
            if None, all variables are plotted.
          num_samples: The number of samples to obtain the error bars
    """      

    
    if feature_ind == None:
      feat_ind_list = [ttt for ttt in range(len(self.feature_names))]
    else:
      feat_ind_list = feature_ind

    feat_res_dict = {}
    for feat_ind in feat_ind_list:
    
      ## Permute the feature
      X_VI = self.X.copy()
      feat_col = X_VI[:, feat_ind].tolist()
      random.shuffle(feat_col)
      X_VI[:, feat_ind] = feat_col

      store_VI = np.zeros(num_samples)
      org_pred = self.gp.sample_y(self.X, n_samples = num_samples)
      permuted_pred = self.gp.sample_y(X_VI, n_samples = num_samples)
      for uuu in range(num_samples):
        VI = self.loss_func(permuted_pred[:,uuu], self.y) - self.loss_func(org_pred[:,uuu], self.y)
        store_VI[uuu] = VI
      m_VI = np.mean(store_VI)
      std_VI = np.std(store_VI)
      feat_res_dict[feat_ind] = [m_VI, std_VI]

    figure, ax = plt.subplots(1,1, figsize = (8,5))

    y_vals = [yyy for yyy in range(len(feat_res_dict))]
    y_name = [self.feature_names[j] for j in feat_res_dict.keys()]
    x_vals = [feat_res_dict[f][0] for f in feat_res_dict.keys()]
    errs = [feat_res_dict[f][1] for f in feat_res_dict.keys()]
    ax.plot(x_vals, y_vals, 'ko')
    ax.set_yticks(y_vals)
    ax.set_yticklabels(y_name)
    ax.grid(axis='y')
    ax.errorbar(x_vals, y_vals, xerr = errs, ecolor='k', fmt='.k', linewidth = 1)
    ax.set_xlabel('VI')

