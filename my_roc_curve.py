def my_roc_curve(model, X, y):
  import my_global_variables
  import matplotlib.pyplot as plt
  import numpy as np
  from sklearn.metrics import roc_curve, auc
  class_name = model.__class__.__name__
  y_proba = model.predict_proba(X)
  proba = y_proba[:, 1]
  fpr, tpr, threshold = roc_curve(y, proba)
  fpr, tpr, threshold
  ##AUC
  roc_auc = auc(fpr, tpr)
  label = f'{class_name} -- {roc_auc}'
  plt.plot(fpr, tpr, c = 'g', label = label, linewidth = 4)
  plt.xlabel('False Positive Rate', fontsize = 16)
  plt.ylabel('True Positive Rate', fontsize = 16)
  plt.title('Receiver Operating Characteristic', fontsize = 16)
  plt.legend(loc = 'lower right', fontsize = 16)
  plt.show()
  my_global_variables.model_roc_auc[class_name] = roc_auc
