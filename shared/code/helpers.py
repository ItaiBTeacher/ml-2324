from cloudpickle import dump, load
from matplotlib.pyplot import axes, close, figure
from numpy import cos, full, nan, pi, ravel
from os.path import isfile
from pandas import DataFrame, Index, Series, Timedelta, Timestamp, concat, to_datetime
from seaborn import scatterplot
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer
from time import time
from typing import Any, Callable

# A do-nothing transformer
identity_transformer = FunctionTransformer((lambda x: x), feature_names_out='one-to-one')

# An arbitrary, but canonical, reference date.
__epoch__ = Timestamp('1970-01-01')

# The function accepts a DataFrame
# all of whose columns are filled with dates,
# truncates them downward to the nearest period ('D' = days, 'M' = months, 'Y' = years),
# and converts them to floats representing
# the number of seconds since epoch.
# Missing values will be represented as numpy.nan's.
def __periods_since_epoch_in_secs__(df: DataFrame, period: str) -> DataFrame:
  #Convert columns to datetime, coercing errors to NaT
  df_dt = df.apply(lambda col: to_datetime(col, utc= True, errors= 'coerce'))
  #Truncate the datetimes to period and then back to timestamp
  df_period = df_dt.apply(lambda col: col.dt.to_period(period).dt.to_timestamp())
  #Calculate the difference in days from the epoch
  df_period_since_epoch = df_period - __epoch__
  #Convert the timedelta to total seconds
  df_seconds_since_epoch = df_period_since_epoch.map( Timedelta.total_seconds,
                                                      na_action='ignore')
  #Convert to float
  return df_seconds_since_epoch.mask(df_seconds_since_epoch.isna(),
                                     nan).astype('float')

def make_date2float_converter(period: str = 'D') -> FunctionTransformer:
  return FunctionTransformer(func=__periods_since_epoch_in_secs__,
                             kw_args={'period': period})

# Function: format_time
# ====================
# Given a time difference in seconds,
# return a string representation thereof, in hh:mm:ss format.
def __format_time__(time: float) -> str:
    hours = int(time / 3600)
    time -= hours * 3600
    minutes = int(time / 60)
    time -= minutes * 60
    seconds = int(round(time))
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

# Function: show_plot
# =====================
# Display a matplotlib Figure object.
# Source: https://stackoverflow.com/a/54579616/1818935
def show_plot(ax: axes) -> None:
    # create a dummy figure and use its
    # manager to display "ax"
    fig = ax.figure
    dummy = figure()
    new_manager = dummy.canvas.manager
    new_manager.canvas.figure = fig
    fig.set_canvas(new_manager.canvas)

# Function: plot_train_test_scores
# ================================
# Given an estimator (clf), and a GridSearchCV object (gs)
# that was initialized with clf as input,
# draw a scatterplot in which a point is drawn
# for every combination of grid parameters.
# The point's coordinates are the corresponding
# cv score w.r.t to the train dataset (x-coordinate)
# and the test score w.r.t. gs' best estimator (y-coordinate).
#
# Parameters:
# gs - A GridSearchCV object. It is assumed that this object's 'scoring' param
#      was explicitly instantiated with a callable, and that no call to
#      'set_params' has ever been invoked on this object.
# clf - The estimator object (possibly a Pipeline, a ColumnTransformer,
#       or a FeatureUnion object) that was used to initialize gs's 'estimator'
#       param.
# scorer - A scorer that takes three arguments: estimator, X, and y_true,
#          and returns a float between 0.0 and 1.0.
#          Cf. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html#sklearn.metrics.make_scorer
# X_train - A non-empty pandas DataFrame.
# y_train - A pandas Series, of the same number of rows as X_train.
# X_test - A non-empty pandas DataFrame with the same set of column names
#          as X_train, and with the same dtypes, respectively.
# y_test - A pandas Series, of the same number of rows as y_test, and of the
#          same dtype as y_train.
#
# Return value: A tuple comprising two elements:
# 1. A matplotlib.Axes object encapsulating the plot.
#    It's the return value of the call to seaborn.scatterplot.
#    Pass it as an argument to the function show_plot, defined below,
#    to display the plot.
# 2. A dictionary comprising two entries:
#    i) 'train': The best grid param combination
#       found by gs.fit(X_train, y_train)
#    ii) 'test': The param combination
#        yielding the best score (per the 'scoring' argument),
#        on X_test and y_test, after the 'clf' object, having been assigned
#        this param combination, has been fitted on X_train and y_train.
def __plot_train_test_scores__(gs, clf, scorer, X_train, X_test, y_train, y_test) \
    -> tuple[axes, dict[str, Any]]:
  # We save clf's initial params,
  # so we can restore them at the end of each loop iteration below.
  base_params = clf.get_params()
  # A table of gs statistics for every combination of grid params.
  results = DataFrame(data= gs.cv_results_)

  # The cross-validation mean score for every combination of grid params.
  cv_scores = results.mean_test_score.to_list()
  # The index of the grid param combination
  # that yielded the highest cv score.
  chosen_params_index = results.rank_test_score.to_list().index(1)

  # A workaround for a bug.
  y_train_rvl = ravel(y_train)
  y_test_rvl = ravel(y_test)

  # For every grid param combination
  # we will set clf's params to this combination,
  # fit clf to X_train and y_train,
  # compute the corresponding score w.r.t. X_test and y_test,
  # and append this score to the following list.
  test_scores = []
  for i in range(results.shape[0]):
    params = results.params[i]
    clf.set_params(**params)
    clf.fit(X_train, y_train_rvl)
    test_score = scorer(clf, X_test, y_test_rvl)
    test_scores.append(test_score)

    # We store the test score
    # corresponding to the best grid param combo.
    if i == chosen_params_index:
      chosen_test_score = test_score

    # We restore clf to its original settings.
    # The double asterisk is a python construct
    # that treats the base_params dictionary
    # as though it was an assignment of arguments to
    # parameters of the function clf.set_params.
    clf.set_params(**base_params)

  best_test_index = test_scores.index(max(test_scores))

  ax = scatterplot(x= cv_scores, y= test_scores)

  # The point corresponding to the best grid param combo
  # will be colored black.
  scatterplot(x= [gs.best_score_],
                  y= [chosen_test_score],
                  color= 'black',
                  ax= ax)

  # The point corresponding to the best test score
  # will be colored red.
  scatterplot(x= [cv_scores[best_test_index]],
                  y= [test_scores[best_test_index]],
                  color='red',
                  ax= ax)

  ax.set_xlabel('Train score')
  ax.set_ylabel('Test score')
  ax.set_title('Train vs. test scores')

  # Suppress the automatic display of the plot.
  # To display it explicitly,
  # use the show_figure function below.
  close()

  best_params = { 'train': results.params[chosen_params_index],
                  'test': results.params[best_test_index]}

  return ax, best_params

def __memoized_action__(action: Callable,
                        file_prefix: str,
                        file_suffix: str = '') -> Any:
  file_base = file_prefix + file_suffix
  target_path = 'data/dynamic/' + file_base + '.cpkl'
  duration_path = 'data/dynamic/' + file_base + '_duration.cpkl'

  if (isfile(target_path)):
    with open(target_path, 'rb') as f:
      target = load(f)

    with open(duration_path, 'rb') as f:
      duration = load(f)

    original = 'original '
  else:
    start = time()
    target = action()
    end = time()
    duration = end - start

    with open(target_path, 'wb') as f:
      dump(target, f)

    with open(duration_path, 'wb') as f:
      dump(duration, f)

    original = ''
  print(original + file_prefix + ' duration: ' + __format_time__(duration))
  return target

def fit(gs: GridSearchCV,
        X_train: DataFrame,
        y_train: Series,
        file_suffix: str = '') -> Any:
  def action() -> GridSearchCV:
    gs.fit(X_train, ravel(y_train))
    return gs
  return __memoized_action__(action, 'fit', file_suffix)

def predict(gs: GridSearchCV,
            X_test: DataFrame,
            file_suffix: str = '') -> Any:
  return __memoized_action__((lambda: gs.predict(X_test)), 'predict', file_suffix)

def plot_train_test_scores( gs: GridSearchCV,
                            clf: BaseEstimator,
                            scorer: Callable,
                            X_train: DataFrame,
                            X_test: DataFrame,
                            y_train: Series,
                            y_test: Series,
                            file_suffix: str = '') -> Any:
  return __memoized_action__((lambda: __plot_train_test_scores__(gs,
                                                                 clf,
                                                                 scorer,
                                                                 X_train,
                                                                 X_test,
                                                                 y_train,
                                                                 y_test)),
                                'plot',
                                file_suffix)


# generate_col_reducing_function_transformer

# Returns a NaN-filled DataFrame of shape (r, c),
# where r == len(index) and c == cols.
# The DataFrame's index is the argument passed as 'index',
# and the column names are 'temp0', 'temp1', etc.
def __nan_df__(index: Index, cols: int) -> DataFrame:
  return DataFrame(data= full(shape= (index.size, cols),
                                 fill_value= nan),
                   columns= ['temp' + str(i) for i in range(cols)],
                   index= index)

# The following function returns a transformer t,
# such that invoking t.fit_transform(df) on a DataFrame df
# returns mapping(df) with the column names as listed in feature_names_out.
# The number of columns returned from mapping(df) should be no greater than
# df's number of columns.
# Why not simply use FunctionTransformer?
# Because FunctionTransformer expects the same number of columns
# in the DataFrame returned by mapping(df)
# as the number of columns in df.
#
# feature_names_in_num should be positive.
# mapping should be a function that accepts a single argument, of type DataFrame,
# whose number of columns is feature_names_in_num,
# and returns a DataFrame whose number of columns is len(feature_names_out).
# len(feature_names_out) should be <= feature_names_in_num.
def make_col_reducing_function_transformer(feature_names_in_num: int,
                                               mapping: Callable,
                                               feature_names_out: list[str]) -> TransformerMixin:
  # The following function ignores its arguments,
  # and returns a list of strings
  # whose length is feature_names_in_num,
  # its head equals feature_names_out,
  # and the rest consists of the strings 'temp0', 'temp1', etc.
  # For example, if feature_names_in_num is 5,
  # and if feature_names_out == ['a', 'b'],
  # then fno)...) will return ['a', 'b', 'temp0', 'temp1', 'temp2'].
  def fno(df: DataFrame, input_feature: list[str]) -> list[str]:
    filler = ['temp' + str(i)
              for i
              in range(feature_names_in_num - len(feature_names_out))]
    return feature_names_out + filler

  # The following function applies to its argument 'df'
  # the mapping passed as argument to 'generate_col_reducing_function_transformer'
  # and appends to the returned DataFrame a number of NaN-filled columns
  # so that the resulting DataFrame has feature_names_in_num number of columns;
  # the extra columns are named 'temp0', 'temp1', etc.
  # For example, if feature_names_in_num == 3,
  # and if result of mapping(df) is the DataFrame
  #   'name'  'age'
  #   =============
  # 0 'John'   17
  # 1 'Liz'    33
  # 2 'Abe'    26
  # then the DataFrame that will be returned from this function will be
  #   'name'  'age' 'temp0'
  #   =====================
  # 0 'John'   17    NaN
  # 1 'Liz'    33    NaN
  # 2 'Abe'    26    NaN
  def func(df: DataFrame) -> DataFrame:
    return concat([mapping(df),
                   __nan_df__(index= df.index,
                              cols= feature_names_in_num - len(feature_names_out))],
                   axis= 1)

  # If df is a DataFrame, then mutate.fit_transform(df) is obtained
  # by appending to the DataFrame returned from mapping(df)
  # a number of NaN-filled columns named 'temp0', 'temp1' etc.,
  # so that the resulting DataFrame has feature_names_in_num number of columns.
  # The names of the returned DataFrame's columns
  # will be the strings in feature_names_out followed by 'temp0', 'temp1', etc.
  # Note that if the 'feature_names_out' parameter were left unassigned,
  # mutate_fit_transform(df) would fail with an error message
  # if df's column names were different
  # than those of the DataFramed returned from func(df).
  mutate = FunctionTransformer(func= func, feature_names_out= fno)

  # If df is a DataFrame, then drop.fit_transform(df) is obtain
  # by keeping the first n columns, and dropping the rest,
  # where n == len(feature_names_out).
  # If verbose_feature_names_out were left unassigned,
  # then the string 'identity_' would be prefixed to the names
  # of the first n columns of the returned DataFrame.
  drop = ColumnTransformer(transformers= [('identity',
                                           identity_transformer,
                                           feature_names_out)],
                            verbose_feature_names_out= False)

  # If the returned Pipeline is saved in the variable p,
  # then rv.fit_transform(df) will return mapping(df),
  # with the column names being those listed in feature_names_out.
  return Pipeline(steps= [('mutate', mutate),
                          ('drop', drop)])

# Cyclic numerical-categorical variables
# ======================================

def __convert_cyclic_xy__(frame: DataFrame, xy: str, cycle_length: int) -> DataFrame:
  mapped = frame.map(lambda val: cos(2 * pi * (val / cycle_length)))
  cols = mapped.columns
  mapped.columns = [col + '_' + xy for col in cols]
  return mapped

def make_cyclic_transformer(cycle_length: int) -> TransformerMixin:
  cycle_x = FunctionTransformer(func=__convert_cyclic_xy__,
                                kw_args={'xy': 'x', 'cycle_length': cycle_length})
  cycle_y = FunctionTransformer(func=__convert_cyclic_xy__,
                                kw_args={'xy': 'y', 'cycle_length': cycle_length})
  return FeatureUnion(transformer_list = [('x', cycle_x), ('y', cycle_y)],
                      verbose_feature_names_out = False)