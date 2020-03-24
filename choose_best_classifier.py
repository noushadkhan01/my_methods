def choose_best_classifier(X, y, C = 1.0, figsize = None, n_neighbors = 5, max_depth = 10,k_fold = 10,
                           svc_kernel = 'rbf', n_components = 2, max_depth_xgb = 4, n_estimators = 10, x_ticks_rotation = -40,
                           plt_show = False, print_results = True, dependent_variable = None, verbose = 0):
  '''This method returns mean and variance of k_fold fold cross validation scores and 
  also returns a boxplot of 10 scores for every model
  
  
  k_fold = default is 10
  
  C = default is 1.0 for Logistic Regression
  n_neighbors = 5 for KNearestNeighbors classifier
  '''
  import matplotlib.pyplot as plt
  import sys
  import numpy as np
  from sklearn import model_selection
  from sklearn.model_selection import cross_val_score
  from sklearn.linear_model import LogisticRegression
  from sklearn.tree import DecisionTreeClassifier
  from sklearn.neighbors import KNeighborsClassifier
  from sklearn.ensemble import RandomForestClassifier
  from xgboost import XGBClassifier
  from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
  from sklearn.decomposition import PCA
  from sklearn.naive_bayes import GaussianNB
  from sklearn.svm import SVC 
  
  #name of dependent variable
  if not dependent_variable:
    try:
      dependent_variable = y.name
    except:
      dependent_variable = None
  # prepare configuration for cross validation test harness
  seed = 7
  # prepare models
  pca = PCA(n_components = n_components)
  X_pca = pca.fit_transform(X)
  models = []
  models.append(('PCA with LR', LogisticRegression(C = C)))
  models.append(('Logistic Regression', LogisticRegression(C = C)))
  models.append(('LDA', LinearDiscriminantAnalysis()))
  models.append(('KNN', KNeighborsClassifier(n_neighbors = n_neighbors)))
  models.append(('DTree', DecisionTreeClassifier(max_depth = max_depth)))
  models.append(('RandomForest', RandomForestClassifier(n_estimators = n_estimators, max_depth = max_depth)))
  models.append(('Niave Bayes', GaussianNB()))
  models.append(('SVM', SVC(kernel = svc_kernel)))
  models.append(('Xgboost', XGBClassifier(max_depth = max_depth_xgb)))
  # evaluate each model in turn
  results = []
  names = []
  scoring = 'accuracy'
  l = len(models)
  n = 1
  if print_results:
    print('model name:-- scores mean, (scores variance)')
  for name, model in models:
    if verbose:
      sys.stdout.write(f'\r running {k_fold} cross validation for {dependent_variable}\'s model No. {n}/{l}')
    kfold = model_selection.KFold(n_splits=k_fold, random_state=seed)
    n += 1
    features = X
    if name == 'PCA with LR':
      features = X_pca
    cv_results = model_selection.cross_val_score(model, features, y, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = f'{name}: -- {cv_results.mean()}, ({cv_results.std()})'
    if verbose:
      sys.stdout.flush()
    if print_results:
      print(msg)
  # boxplot algorithm comparison
  title = f'Algorithm Comaprision for {dependent_variable}'
  sys.stdout.write(f'\r Done for {dependent_variable}\'s model')
  if figsize:
    plt.figure(figsize = figsize)
  plt.title(title)
  plt.boxplot(results)
  ypos = np.arange(1, len(names)+1)
  plt.xticks(ypos, names, rotation = x_ticks_rotation)
  sys.stdout.write('\r Done')
  if plt_show:
    plt.show()
