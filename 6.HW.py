import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LinearRegression, Lasso, LassoCV
from sklearn.ensemble import RandomForestRegressor


data = pd.read_csv('dataset/winequality-white.csv', sep=';')

y = data['quality']
data.drop('quality', axis=1, inplace=True)

X_train, X_holdout, y_train, y_holdout = train_test_split(data, y, test_size=0.3, random_state=17)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_holdout_scaled = scaler.transform(X_holdout)


linreg = LinearRegression()
linreg.fit(X_train_scaled, y_train)

"""Вопрос 1. Каковы среднеквадратичные ошибки линейной регрессии на обучающей и отложенной выборках?"""
print("Mean squared error (train): %.3f" % mean_squared_error(y_train, linreg.predict(X_train_scaled)))
print("Mean squared error (test): %.3f" % mean_squared_error(y_holdout, linreg.predict(X_holdout_scaled)))


"""Вопрос 2. Какой признак линейная регрессия считает наиболее сильно влияющим на качество вина?"""
linreg_coef = pd.DataFrame({'coef': linreg.coef_}, index=X_train.columns)
linreg_coef.sort_values(by='coef', inplace=True)


lasso1 = Lasso(alpha=0.01, random_state=17)
lasso1.fit(X_train_scaled, y_train)

lasso1_coef = pd.DataFrame({'coef': lasso1.coef_}, index=X_train.columns)
lasso1_coef.sort_values(by='coef', inplace=True)

alphas = np.logspace(-6, 2, 200)
lasso_cv = LassoCV(alphas=alphas, cv=5, random_state=17)
lasso_cv.fit(X_train_scaled, y_train)


"""Вопрос 3. Какой признак "обнулился первым" в настроенной модели LASSO?"""
lasso_cv_coef = pd.DataFrame({'coef': lasso_cv.coef_}, index=X_train.columns)
lasso_cv_coef.sort_values(by='coef', inplace=True)

print("Mean squared error (train): %.3f" % mean_squared_error(y_train, lasso_cv.predict(X_train_scaled)))
print("Mean squared error (test): %.3f" % mean_squared_error(y_holdout, lasso_cv.predict(X_holdout_scaled)))


forest = RandomForestRegressor(random_state=17)
forest.fit(X_train, y_train)

print("Mean squared error (train): %.3f" % mean_squared_error(y_train, forest.predict(X_train)))
print("Forest Mean squared error (cv): %.3f" % cross_val_score(forest, X_train, y_train, scoring='neg_mean_squared_error').mean())
print("Mean squared error (test): %.3f" % mean_squared_error(y_holdout, forest.predict(X_holdout)))

#
# forest_params = {'max_depth': list(range(10, 25)),
#                  'min_samples_leaf': list(range(1, 8)),
#                  'max_features': list(range(6, 12))}
#
# locally_best_forest = GridSearchCV(forest, param_grid=forest_params)
# locally_best_forest.fit(X_train, y_train)
# print(locally_best_forest.best_params_, locally_best_forest.best_score_)
#

upgrade_forest = RandomForestRegressor(max_depth=19, max_features=7, min_samples_leaf=1)

upgrade_forest.fit(X_train, y_train)

print("Mean squared error (cv): %.3f" % cross_val_score(upgrade_forest, X_train, y_train, scoring='neg_mean_squared_error').mean())
print("Mean squared error (test): %.3f" % mean_squared_error(y_holdout, upgrade_forest.predict(X_holdout)))


"""Вопрос 7: Какой признак оказался главным в настроенной модели случайного леса?"""
rf_importance = pd.DataFrame({'f_i': upgrade_forest.feature_importances_}, index=X_train.columns)
rf_importance.sort_values(by='f_i', inplace=True)

