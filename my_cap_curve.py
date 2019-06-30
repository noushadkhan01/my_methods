def my_cap_curve(model, X, y):
  import matplotlib.pyplot as plt
  import numpy as np
  import my_global_variables
  from sklearn.metrics import roc_curve, auc
  class_name = model.__class__.__name__
  total = len(y)
  class_1_count = np.sum(y)
  class_0_count = total - class_1_count
  probs = model.predict_proba(X)
  probs = probs[:, 1]
  model_y = [y for _, y in sorted(zip(probs, y), reverse = True)]
  y_values = np.append([0], np.cumsum(model_y))
  x_values = np.arange(0, total + 1)
  plt.plot([0, total], [0, class_1_count], c = 'r', linestyle = '--', label = 'Random Model')
  plt.plot([0, class_1_count, total], [0, class_1_count, class_1_count], c = 'grey', linewidth = 2, label = 'Perfect Model')
  plt.plot(x_values, y_values, c = 'b', label = f'{class_name} Classifier', linewidth = 4)
  plt.xlabel('Total observations', fontsize = 16)
  plt.ylabel('Class 1 observations', fontsize = 16)
  plt.title('Cumulative Accuracy Profile', fontsize = 16)
  plt.legend(loc = 'lower right', fontsize = 16)
  plt.show()
  # Area under Random Model
  a = auc([0, total], [0, class_1_count])

  # Area between Perfect and Random Model
  aP = auc([0, class_1_count, total], [0, class_1_count, class_1_count]) - a

  # Area between Trained and Random Model
  aR = auc(x_values, y_values) - a

  print("Accuracy Rate for {class_name} Classifier: {}".format(aR / aP))
  my_global_variables.model_cap_scores[class_name] = aR/aP
