def my_roc_curve(model, X, y, fig_size = (10, 5),legend_font_size = 3,loc = 'best', linewidth = 2,label_font_size = 4, poly_features = False):
  import my_global_variables
  import matplotlib.pyplot as plt
  import numpy as np
  from sklearn.metrics import roc_curve, auc
  class_name = model.__class__.__name__
  if poly_features:
    class_name += '_poly'
  y_proba = model.predict_proba(X)
  proba = y_proba[:, 1]
  fpr, tpr, threshold = roc_curve(y, proba)
  fpr, tpr, threshold
  ##AUC
  roc_auc = auc(fpr, tpr)
  label = f'{class_name} -- {roc_auc}'
  plt.plot(fpr, tpr, c = 'g', label = label, linewidth = linewidth)
  plt.xlabel('False Positive Rate', fontsize = label_font_size)
  plt.ylabel('True Positive Rate', fontsize = label_font_size)
  plt.title('Receiver Operating Characteristic', fontsize = label_font_size)
  plt.legend(loc = loc, fontsize = legend_font_size)
  plt.show()
  my_global_variables.model_roc_auc[class_name] = roc_auc
