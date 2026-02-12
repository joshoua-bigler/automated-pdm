import numpy as np
import matplotlib.pyplot as plt


def scatter_plot(pred: np.ndarray, target: np.ndarray, rmse: float) -> plt.Figure:
  ''' Scatter plot of predicted vs true RUL values with RMSE in title

      Parameters
      ----------
      pred : 
          Predicted RUL values.
      target : 
          True RUL values.
      rmse :
          Root Mean Square Error between predicted and true RUL values.
  '''
  fig, ax = plt.subplots(figsize=(6, 6))
  ax.scatter(target, pred, s=12, alpha=0.45, label='samples')
  lims = [min(target.min(), pred.min()), max(target.max(), pred.max())]
  ax.plot(lims, lims, 'k--', lw=1, label='ideal')
  ax.set(xlabel='True RUL', ylabel='Estimated RUL', title=f'Val RMSE = {rmse:.2f}')
  ax.legend()
  plt.grid()
  return fig
