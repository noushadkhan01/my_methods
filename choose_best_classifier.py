def choose_best_classifier(X, y, C = 1.0,figsize = (10, 5), n_neighbors = 5, max_depth = 10, svc_kernel = 'rbf', n_components = 2, max_depth_xgb = 4, n_estimators = 10, x_ticks_rotation = -40):
  import matplotlib.pyplot as plt
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
  # prepare configuration for cross validation test harness
  seed = 7
  # prepare models
  pca = PCA(n_components = n_components)
  X_pca = pca.fit_transform(X)
  models = []
  models.append(('PCA with LR', LogisticRegression(C = C)))
  models.append(('LR', LogisticRegression(C = C)))
  models.append(('LDA', LinearDiscriminantAnalysis()))
  models.append(('KNN', KNeighborsClassifier(n_neighbors = n_neighbors)))
  models.append(('DTree', DecisionTreeClassifier(max_depth = max_depth)))
  models.append(('RandomForest', RandomForestClassifier(n_estimators = n_estimators, max_depth = max_depth)))
  models.append(('NB', GaussianNB()))
  models.append(('SVM', SVC(kernel = svc_kernel)))
  models.append(('Xgboost', XGBClassifier(max_depth = max_depth_xgb)))
  # evaluate each model in turn
  results = []
  names = []
  scoring = 'accuracy'
  for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    features = X
    if name == 'PCA with LR':
      features = X_pca
    cv_results = model_selection.cross_val_score(model, features, y, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
  # boxplot algorithm comparison
  fig = plt.figure(figsize = figsize)
  fig.suptitle('Algorithm Comparison')
  ax = fig.add_subplot(111)
  plt.boxplot(results)
  ax.set_xticklabels(names)
  plt.xticks(rotation = x_ticks_rotation)
  #plt.show()
