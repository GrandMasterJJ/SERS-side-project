{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# SVM for SERS data analysing"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "from sklearn import model_selection\n",
    "from sklearn import svm"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "from csv import reader\n",
    "with open ('210924_bw25113_30min_MLdataset.csv') as f:\n",
    "    csv_reader = reader(f)\n",
    "    csv_headings = next(csv_reader)\n",
    "\n",
    "    features = csv_headings[1:]\n",
    "    #print(csv_headings)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.read_csv('copy - 210924_bw25113_30min_MLdataset.csv',names=csv_headings)\n",
    "#iris_features = ['sepal-len', 'sepal-width', 'petal-len', 'petal-width']\n",
    "\n",
    "# Extract features\n",
    "X = df.loc[:, features].values\n",
    "\n",
    "# Extract target i.e. species\n",
    "Y = df.loc[:, ['Wavenumber(cm^-1)']].values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "# keeping 40% reserved for testing  and 60% data will be used to train and form model.\n",
    "X_train, X_test, Y_train, Y_test = model_selection.train_test_split (X, Y, test_size=0.4, random_state=0)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "E:\\anaconda3\\lib\\site-packages\\sklearn\\utils\\validation.py:72: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().\n",
      "  return f(**kwargs)\n"
     ]
    }
   ],
   "source": [
    "# Build an SVC (Support Vector Classification) model using linear regression\n",
    "svc_model = svm.SVC(kernel='linear', C=1).fit(X_train, Y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.9\n"
     ]
    }
   ],
   "source": [
    "print(svc_model.score(X_test, Y_test))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "E:\\anaconda3\\lib\\site-packages\\sklearn\\utils\\validation.py:72: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().\n",
      "  return f(**kwargs)\n",
      "E:\\anaconda3\\lib\\site-packages\\sklearn\\utils\\validation.py:72: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().\n",
      "  return f(**kwargs)\n",
      "E:\\anaconda3\\lib\\site-packages\\sklearn\\utils\\validation.py:72: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().\n",
      "  return f(**kwargs)\n",
      "E:\\anaconda3\\lib\\site-packages\\sklearn\\utils\\validation.py:72: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().\n",
      "  return f(**kwargs)\n",
      "E:\\anaconda3\\lib\\site-packages\\sklearn\\utils\\validation.py:72: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().\n",
      "  return f(**kwargs)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[0.86875 0.91875 0.90625 0.95625 0.85625]\n",
      "0.9012500000000001\n"
     ]
    }
   ],
   "source": [
    "scores_res = model_selection.cross_val_score(svc_model, X, Y, cv=5)\n",
    "\n",
    "# Print the accuracy of each fold \n",
    "print(scores_res)\n",
    "\n",
    "# And the mean accuracy of all 5 folds.\n",
    "print(scores_res.mean())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
