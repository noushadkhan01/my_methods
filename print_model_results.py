def print_model_results(X_train, X_test,y_train, y_test, model, variable = False):
  from sklearn.metrics import confusion_matrix, classification_report
  if variable:
    global model_score
    global model_cm
  from sklearn.metrics import confusion_matrix, classification_report
  model.fit(X_train, y_train)
  ts = model.score(X_train, y_train)
  print(F' Train score is {ts}')
  print('\n')
  tst = model.score(X_test, y_test)
  print(f'Test score is {tst}')
  print('\n\n')
  y_pred = model.predict(X_test)
  tcm = confusion_matrix(y_train, model.predict(X_train))
  print(f'Train confusion matrix is \n {tcm}\n')
  ttcm = confusion_matrix(y_test, y_pred)
  print(f'Test confusion matrix is \n {ttcm}')
  print('\n\n')
  print(f'Test Set classification report is \n {classification_report(y_test, y_pred)}')
  class_name = model.__class__.__name__
  if variable:
    model_score[class_name] = {'train':ts, 'test': tst}
    model_cm[class_name] = {'train':tcm, 'test':ttcm}
  return model
