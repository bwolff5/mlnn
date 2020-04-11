{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Assignment 5\n",
    "\n",
    "1. Choose a regression dataset (bikeshare is allowed), perform a test/train split, and build a regression model (just like in assingnment 3), and calculate the \n",
    "    + Training Error (MSE, MAE)\n",
    "    + Testing Error (MSE, MAE)\n",
    "2. Choose a classification dataset (not the adult.data set, The UCI repository has many datasets as well as Kaggle), perform test/train split and create a classification model (your choice but DecisionTree is fine). Calculate \n",
    "    + Accuracy\n",
    "    + Confusion Matrix\n",
    "    + Classifcation Report\n",
    "    \n",
    "3. (Bonus) See if you can improve the classification model's performance with any tricks you can think of (modify features, remove features, polynomial features)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 584,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.linear_model import LinearRegression\n",
    "import numpy as np\n",
    "import matplotlib.pylab as plt\n",
    "%matplotlib inline\n",
    "from sklearn import linear_model\n",
    "import sklearn as sk\n",
    "from sklearn.model_selection import train_test_split"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Question 1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 585,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Unnamed: 0</th>\n",
       "      <th>Age</th>\n",
       "      <th>Sex</th>\n",
       "      <th>ChestPain</th>\n",
       "      <th>RestBP</th>\n",
       "      <th>Chol</th>\n",
       "      <th>Fbs</th>\n",
       "      <th>RestECG</th>\n",
       "      <th>MaxHR</th>\n",
       "      <th>ExAng</th>\n",
       "      <th>Oldpeak</th>\n",
       "      <th>Slope</th>\n",
       "      <th>Ca</th>\n",
       "      <th>Thal</th>\n",
       "      <th>AHD</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>63</td>\n",
       "      <td>1</td>\n",
       "      <td>typical</td>\n",
       "      <td>145</td>\n",
       "      <td>233</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>150</td>\n",
       "      <td>0</td>\n",
       "      <td>2.3</td>\n",
       "      <td>3</td>\n",
       "      <td>0.0</td>\n",
       "      <td>fixed</td>\n",
       "      <td>No</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>67</td>\n",
       "      <td>1</td>\n",
       "      <td>asymptomatic</td>\n",
       "      <td>160</td>\n",
       "      <td>286</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>108</td>\n",
       "      <td>1</td>\n",
       "      <td>1.5</td>\n",
       "      <td>2</td>\n",
       "      <td>3.0</td>\n",
       "      <td>normal</td>\n",
       "      <td>Yes</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>2</td>\n",
       "      <td>3</td>\n",
       "      <td>67</td>\n",
       "      <td>1</td>\n",
       "      <td>asymptomatic</td>\n",
       "      <td>120</td>\n",
       "      <td>229</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>129</td>\n",
       "      <td>1</td>\n",
       "      <td>2.6</td>\n",
       "      <td>2</td>\n",
       "      <td>2.0</td>\n",
       "      <td>reversable</td>\n",
       "      <td>Yes</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>3</td>\n",
       "      <td>4</td>\n",
       "      <td>37</td>\n",
       "      <td>1</td>\n",
       "      <td>nonanginal</td>\n",
       "      <td>130</td>\n",
       "      <td>250</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>187</td>\n",
       "      <td>0</td>\n",
       "      <td>3.5</td>\n",
       "      <td>3</td>\n",
       "      <td>0.0</td>\n",
       "      <td>normal</td>\n",
       "      <td>No</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>4</td>\n",
       "      <td>5</td>\n",
       "      <td>41</td>\n",
       "      <td>0</td>\n",
       "      <td>nontypical</td>\n",
       "      <td>130</td>\n",
       "      <td>204</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>172</td>\n",
       "      <td>0</td>\n",
       "      <td>1.4</td>\n",
       "      <td>1</td>\n",
       "      <td>0.0</td>\n",
       "      <td>normal</td>\n",
       "      <td>No</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>298</td>\n",
       "      <td>299</td>\n",
       "      <td>45</td>\n",
       "      <td>1</td>\n",
       "      <td>typical</td>\n",
       "      <td>110</td>\n",
       "      <td>264</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>132</td>\n",
       "      <td>0</td>\n",
       "      <td>1.2</td>\n",
       "      <td>2</td>\n",
       "      <td>0.0</td>\n",
       "      <td>reversable</td>\n",
       "      <td>Yes</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>299</td>\n",
       "      <td>300</td>\n",
       "      <td>68</td>\n",
       "      <td>1</td>\n",
       "      <td>asymptomatic</td>\n",
       "      <td>144</td>\n",
       "      <td>193</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>141</td>\n",
       "      <td>0</td>\n",
       "      <td>3.4</td>\n",
       "      <td>2</td>\n",
       "      <td>2.0</td>\n",
       "      <td>reversable</td>\n",
       "      <td>Yes</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>300</td>\n",
       "      <td>301</td>\n",
       "      <td>57</td>\n",
       "      <td>1</td>\n",
       "      <td>asymptomatic</td>\n",
       "      <td>130</td>\n",
       "      <td>131</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>115</td>\n",
       "      <td>1</td>\n",
       "      <td>1.2</td>\n",
       "      <td>2</td>\n",
       "      <td>1.0</td>\n",
       "      <td>reversable</td>\n",
       "      <td>Yes</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>301</td>\n",
       "      <td>302</td>\n",
       "      <td>57</td>\n",
       "      <td>0</td>\n",
       "      <td>nontypical</td>\n",
       "      <td>130</td>\n",
       "      <td>236</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>174</td>\n",
       "      <td>0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>2</td>\n",
       "      <td>1.0</td>\n",
       "      <td>normal</td>\n",
       "      <td>Yes</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>302</td>\n",
       "      <td>303</td>\n",
       "      <td>38</td>\n",
       "      <td>1</td>\n",
       "      <td>nonanginal</td>\n",
       "      <td>138</td>\n",
       "      <td>175</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>173</td>\n",
       "      <td>0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1</td>\n",
       "      <td>NaN</td>\n",
       "      <td>normal</td>\n",
       "      <td>No</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>303 rows × 15 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     Unnamed: 0  Age  Sex     ChestPain  RestBP  Chol  Fbs  RestECG  MaxHR  \\\n",
       "0             1   63    1       typical     145   233    1        2    150   \n",
       "1             2   67    1  asymptomatic     160   286    0        2    108   \n",
       "2             3   67    1  asymptomatic     120   229    0        2    129   \n",
       "3             4   37    1    nonanginal     130   250    0        0    187   \n",
       "4             5   41    0    nontypical     130   204    0        2    172   \n",
       "..          ...  ...  ...           ...     ...   ...  ...      ...    ...   \n",
       "298         299   45    1       typical     110   264    0        0    132   \n",
       "299         300   68    1  asymptomatic     144   193    1        0    141   \n",
       "300         301   57    1  asymptomatic     130   131    0        0    115   \n",
       "301         302   57    0    nontypical     130   236    0        2    174   \n",
       "302         303   38    1    nonanginal     138   175    0        0    173   \n",
       "\n",
       "     ExAng  Oldpeak  Slope   Ca        Thal  AHD  \n",
       "0        0      2.3      3  0.0       fixed   No  \n",
       "1        1      1.5      2  3.0      normal  Yes  \n",
       "2        1      2.6      2  2.0  reversable  Yes  \n",
       "3        0      3.5      3  0.0      normal   No  \n",
       "4        0      1.4      1  0.0      normal   No  \n",
       "..     ...      ...    ...  ...         ...  ...  \n",
       "298      0      1.2      2  0.0  reversable  Yes  \n",
       "299      0      3.4      2  2.0  reversable  Yes  \n",
       "300      1      1.2      2  1.0  reversable  Yes  \n",
       "301      0      0.0      2  1.0      normal  Yes  \n",
       "302      0      0.0      1  NaN      normal   No  \n",
       "\n",
       "[303 rows x 15 columns]"
      ]
     },
     "execution_count": 585,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df = pd.read_csv('../data/Heart.csv', index_col=False)\n",
    "df"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Setup\n",
    "\n",
    "Using heart dataset. Independent variable is age. Dependent variable is resting blood pressure"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 586,
   "metadata": {},
   "outputs": [],
   "source": [
    "x = df[\"Age\"]\n",
    "y = df[\"RestBP\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 587,
   "metadata": {},
   "outputs": [],
   "source": [
    "x = x.values.reshape(-1, 1)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Standard Linear Regression"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 588,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)"
      ]
     },
     "execution_count": 588,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model = LinearRegression()\n",
    "model.fit(x, y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 589,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(array([0.55483611]), 101.48507719035973)"
      ]
     },
     "execution_count": 589,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.coef_, model.intercept_"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 590,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[<matplotlib.lines.Line2D at 0x1378a7a7088>]"
      ]
     },
     "execution_count": 590,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.scatter(x,y)\n",
    "plt.plot(x, np.dot(x, model.coef_) + model.intercept_)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Split data into training and test"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 591,
   "metadata": {},
   "outputs": [],
   "source": [
    "x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.5)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Create model using training dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 592,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(array([0.43600972]), 107.17024595654046)"
      ]
     },
     "execution_count": 592,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model = LinearRegression()\n",
    "model.fit(x_train, y_train)\n",
    "model.coef_, model.intercept_"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Plot results of training dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 593,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[<matplotlib.lines.Line2D at 0x1378a7a6908>]"
      ]
     },
     "execution_count": 593,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.scatter(x,y)\n",
    "plt.plot(x, np.dot(x, model.coef_) + model.intercept_)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### MSE for train model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 594,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "269.1081748253961"
      ]
     },
     "execution_count": 594,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "mse_train = sk.metrics.mean_squared_error(y_train, np.dot(x_train, model.coef_) + model.intercept_)\n",
    "mse_train"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### MAE for Train Model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 595,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "12.657461408026133"
      ]
     },
     "execution_count": 595,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "mae_train = sk.metrics.mean_absolute_error(y_train, np.dot(x_train, model.coef_) + model.intercept_)\n",
    "mae_train"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Plot model on test data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 596,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[<matplotlib.lines.Line2D at 0x1378a83e348>]"
      ]
     },
     "execution_count": 596,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.scatter(x_test,y_test)\n",
    "plt.plot(x_test, np.dot(x_test, model.coef_) + model.intercept_)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### MSE for test model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 557,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "298.90066391421675"
      ]
     },
     "execution_count": 557,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "mse_test = sk.metrics.mean_squared_error(y_test, np.dot(x_test, model.coef_) + model.intercept_)\n",
    "mse_test"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### MAE for Test Model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 558,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "13.640250500443702"
      ]
     },
     "execution_count": 558,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "mae_test = sk.metrics.mean_absolute_error(y_test, np.dot(x_test, model.coef_) + model.intercept_)\n",
    "mae_test"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### All Metrics"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 573,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Train MSE:  275.9703031686645\n",
      "Train MAE:  12.50621640046854\n",
      "Test MSE:  298.90066391421675\n",
      "Test MAE:  13.640250500443702\n"
     ]
    }
   ],
   "source": [
    "print(\"Train MSE: \", mse_train)\n",
    "print(\"Train MAE: \", mae_train)\n",
    "print(\"Test MSE: \", mse_test)\n",
    "print(\"Test MAE: \", mae_test)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Question 2"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Choose a classification dataset (not the adult.data set, The UCI repository has many datasets as well as Kaggle), perform test/train split and create a classification model (your choice but DecisionTree is fine). Calculate: \n",
    "+ Accuracy\n",
    "+ Confusion Matrix\n",
    "+ Classifcation Report"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Read in credit dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 599,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.read_csv(\"../data/Credit.csv\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Change rating to logical. 1 for score 450 and over, 0 if not."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 600,
   "metadata": {},
   "outputs": [],
   "source": [
    "df[[\"Rating\"]] = df[[\"Rating\"]] >= 450\n",
    "df[[\"Rating\"]] = df[[\"Rating\"]].astype(int)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Identify columns for removal and remove"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 601,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['Unnamed: 0', 'Income', 'Limit', 'Rating', 'Cards', 'Age', 'Education',\n",
       "       'Gender', 'Student', 'Married', 'Ethnicity', 'Balance'],\n",
       "      dtype='object')"
      ]
     },
     "execution_count": 601,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 602,
   "metadata": {},
   "outputs": [],
   "source": [
    "to_remove = ['Unnamed: 0','Gender', 'Student', 'Married', 'Ethnicity']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 603,
   "metadata": {},
   "outputs": [],
   "source": [
    "x = df.copy().drop(to_remove, axis=1)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Split dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 604,
   "metadata": {},
   "outputs": [],
   "source": [
    "x_train, x_test = train_test_split(x, test_size=.5)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Import decision tree function and run model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 605,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.tree import DecisionTreeClassifier\n",
    "model = DecisionTreeClassifier(criterion='entropy')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 606,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=None,\n",
       "                       max_features=None, max_leaf_nodes=None,\n",
       "                       min_impurity_decrease=0.0, min_impurity_split=None,\n",
       "                       min_samples_leaf=1, min_samples_split=2,\n",
       "                       min_weight_fraction_leaf=0.0, presort=False,\n",
       "                       random_state=None, splitter='best')"
      ]
     },
     "execution_count": 606,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.fit(x_train.drop([\"Rating\"], axis = 1), x_train.Rating)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Display Feature Importances"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 628,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[('Income', 0.0),\n",
       " ('Limit', 0.9417065210990727),\n",
       " ('Cards', 0.0),\n",
       " ('Age', 0.018343122791938813),\n",
       " ('Education', 0.02160723331704986),\n",
       " ('Balance', 0.018343122791938813)]"
      ]
     },
     "execution_count": 628,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "list(zip(x_train.drop([\"Rating\"], axis = 1).columns, model.feature_importances_))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Import Decision Tree Metrics, create predictions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 629,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import (accuracy_score, \n",
    "                             classification_report, \n",
    "                             confusion_matrix, auc, roc_curve\n",
    "                            )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 638,
   "metadata": {},
   "outputs": [],
   "source": [
    "predictions_train = model.predict(x_train.drop([\"Balance\"], axis=1))\n",
    "predictions_test = model.predict(x_test.drop([\"Balance\"], axis=1))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Training model metrics\n",
    "*Accuracy Score*"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 637,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.965"
      ]
     },
     "execution_count": 637,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "accuracy_score(x_train.Rating, predictions_train)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "*Confusion Matrix*"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 641,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[154,   3],\n",
       "       [  4,  39]], dtype=int64)"
      ]
     },
     "execution_count": 641,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "sk.metrics.confusion_matrix(x_train.Rating, predictions_train)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "*Classification Report*"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 642,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.97      0.98      0.98       157\n",
      "           1       0.93      0.91      0.92        43\n",
      "\n",
      "    accuracy                           0.96       200\n",
      "   macro avg       0.95      0.94      0.95       200\n",
      "weighted avg       0.96      0.96      0.96       200\n",
      "\n"
     ]
    }
   ],
   "source": [
    "print(classification_report(x_train.Rating, predictions_train))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Testing Model Metrics\n",
    "*Accuracy Score*"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 636,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.975"
      ]
     },
     "execution_count": 636,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "accuracy_score(x_test.Rating, predictions_test)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "*Confusion Matrix*"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 640,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[148,   3],\n",
       "       [  2,  47]], dtype=int64)"
      ]
     },
     "execution_count": 640,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "sk.metrics.confusion_matrix(x_test.Rating, predictions_test)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "*Classification Report*"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 643,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.99      0.98      0.98       151\n",
      "           1       0.94      0.96      0.95        49\n",
      "\n",
      "    accuracy                           0.97       200\n",
      "   macro avg       0.96      0.97      0.97       200\n",
      "weighted avg       0.98      0.97      0.98       200\n",
      "\n"
     ]
    }
   ],
   "source": [
    "print(classification_report(x_test.Rating, predictions_test))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Question 3"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "(Bonus) See if you can improve the classification model's performance with any tricks you can think of (modify features, remove features, polynomial features)"
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
   "execution_count": 744,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>0</th>\n",
       "      <th>1</th>\n",
       "      <th>2</th>\n",
       "      <th>3</th>\n",
       "      <th>4</th>\n",
       "      <th>5</th>\n",
       "      <th>6</th>\n",
       "      <th>7</th>\n",
       "      <th>8</th>\n",
       "      <th>9</th>\n",
       "      <th>10</th>\n",
       "      <th>11</th>\n",
       "      <th>12</th>\n",
       "      <th>13</th>\n",
       "      <th>14</th>\n",
       "      <th>15</th>\n",
       "      <th>16</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <td>0</td>\n",
       "      <td>republican</td>\n",
       "      <td>n</td>\n",
       "      <td>y</td>\n",
       "      <td>n</td>\n",
       "      <td>y</td>\n",
       "      <td>y</td>\n",
       "      <td>y</td>\n",
       "      <td>n</td>\n",
       "      <td>n</td>\n",
       "      <td>n</td>\n",
       "      <td>y</td>\n",
       "      <td>?</td>\n",
       "      <td>y</td>\n",
       "      <td>y</td>\n",
       "      <td>y</td>\n",
       "      <td>n</td>\n",
       "      <td>y</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>1</td>\n",
       "      <td>republican</td>\n",
       "      <td>n</td>\n",
       "      <td>y</td>\n",
       "      <td>n</td>\n",
       "      <td>y</td>\n",
       "      <td>y</td>\n",
       "      <td>y</td>\n",
       "      <td>n</td>\n",
       "      <td>n</td>\n",
       "      <td>n</td>\n",
       "      <td>n</td>\n",
       "      <td>n</td>\n",
       "      <td>y</td>\n",
       "      <td>y</td>\n",
       "      <td>y</td>\n",
       "      <td>n</td>\n",
       "      <td>?</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>2</td>\n",
       "      <td>democrat</td>\n",
       "      <td>?</td>\n",
       "      <td>y</td>\n",
       "      <td>y</td>\n",
       "      <td>?</td>\n",
       "      <td>y</td>\n",
       "      <td>y</td>\n",
       "      <td>n</td>\n",
       "      <td>n</td>\n",
       "      <td>n</td>\n",
       "      <td>n</td>\n",
       "      <td>y</td>\n",
       "      <td>n</td>\n",
       "      <td>y</td>\n",
       "      <td>y</td>\n",
       "      <td>n</td>\n",
       "      <td>n</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>3</td>\n",
       "      <td>democrat</td>\n",
       "      <td>n</td>\n",
       "      <td>y</td>\n",
       "      <td>y</td>\n",
       "      <td>n</td>\n",
       "      <td>?</td>\n",
       "      <td>y</td>\n",
       "      <td>n</td>\n",
       "      <td>n</td>\n",
       "      <td>n</td>\n",
       "      <td>n</td>\n",
       "      <td>y</td>\n",
       "      <td>n</td>\n",
       "      <td>y</td>\n",
       "      <td>n</td>\n",
       "      <td>n</td>\n",
       "      <td>y</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>4</td>\n",
       "      <td>democrat</td>\n",
       "      <td>y</td>\n",
       "      <td>y</td>\n",
       "      <td>y</td>\n",
       "      <td>n</td>\n",
       "      <td>y</td>\n",
       "      <td>y</td>\n",
       "      <td>n</td>\n",
       "      <td>n</td>\n",
       "      <td>n</td>\n",
       "      <td>n</td>\n",
       "      <td>y</td>\n",
       "      <td>?</td>\n",
       "      <td>y</td>\n",
       "      <td>y</td>\n",
       "      <td>y</td>\n",
       "      <td>y</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>430</td>\n",
       "      <td>republican</td>\n",
       "      <td>n</td>\n",
       "      <td>n</td>\n",
       "      <td>y</td>\n",
       "      <td>y</td>\n",
       "      <td>y</td>\n",
       "      <td>y</td>\n",
       "      <td>n</td>\n",
       "      <td>n</td>\n",
       "      <td>y</td>\n",
       "      <td>y</td>\n",
       "      <td>n</td>\n",
       "      <td>y</td>\n",
       "      <td>y</td>\n",
       "      <td>y</td>\n",
       "      <td>n</td>\n",
       "      <td>y</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>431</td>\n",
       "      <td>democrat</td>\n",
       "      <td>n</td>\n",
       "      <td>n</td>\n",
       "      <td>y</td>\n",
       "      <td>n</td>\n",
       "      <td>n</td>\n",
       "      <td>n</td>\n",
       "      <td>y</td>\n",
       "      <td>y</td>\n",
       "      <td>y</td>\n",
       "      <td>y</td>\n",
       "      <td>n</td>\n",
       "      <td>n</td>\n",
       "      <td>n</td>\n",
       "      <td>n</td>\n",
       "      <td>n</td>\n",
       "      <td>y</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>432</td>\n",
       "      <td>republican</td>\n",
       "      <td>n</td>\n",
       "      <td>?</td>\n",
       "      <td>n</td>\n",
       "      <td>y</td>\n",
       "      <td>y</td>\n",
       "      <td>y</td>\n",
       "      <td>n</td>\n",
       "      <td>n</td>\n",
       "      <td>n</td>\n",
       "      <td>n</td>\n",
       "      <td>y</td>\n",
       "      <td>y</td>\n",
       "      <td>y</td>\n",
       "      <td>y</td>\n",
       "      <td>n</td>\n",
       "      <td>y</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>433</td>\n",
       "      <td>republican</td>\n",
       "      <td>n</td>\n",
       "      <td>n</td>\n",
       "      <td>n</td>\n",
       "      <td>y</td>\n",
       "      <td>y</td>\n",
       "      <td>y</td>\n",
       "      <td>?</td>\n",
       "      <td>?</td>\n",
       "      <td>?</td>\n",
       "      <td>?</td>\n",
       "      <td>n</td>\n",
       "      <td>y</td>\n",
       "      <td>y</td>\n",
       "      <td>y</td>\n",
       "      <td>n</td>\n",
       "      <td>y</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>434</td>\n",
       "      <td>republican</td>\n",
       "      <td>n</td>\n",
       "      <td>y</td>\n",
       "      <td>n</td>\n",
       "      <td>y</td>\n",
       "      <td>y</td>\n",
       "      <td>y</td>\n",
       "      <td>n</td>\n",
       "      <td>n</td>\n",
       "      <td>n</td>\n",
       "      <td>y</td>\n",
       "      <td>n</td>\n",
       "      <td>y</td>\n",
       "      <td>y</td>\n",
       "      <td>y</td>\n",
       "      <td>?</td>\n",
       "      <td>n</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>435 rows × 17 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "             0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16\n",
       "0    republican  n  y  n  y  y  y  n  n  n  y  ?  y  y  y  n  y\n",
       "1    republican  n  y  n  y  y  y  n  n  n  n  n  y  y  y  n  ?\n",
       "2      democrat  ?  y  y  ?  y  y  n  n  n  n  y  n  y  y  n  n\n",
       "3      democrat  n  y  y  n  ?  y  n  n  n  n  y  n  y  n  n  y\n",
       "4      democrat  y  y  y  n  y  y  n  n  n  n  y  ?  y  y  y  y\n",
       "..          ... .. .. .. .. .. .. .. .. .. .. .. .. .. .. .. ..\n",
       "430  republican  n  n  y  y  y  y  n  n  y  y  n  y  y  y  n  y\n",
       "431    democrat  n  n  y  n  n  n  y  y  y  y  n  n  n  n  n  y\n",
       "432  republican  n  ?  n  y  y  y  n  n  n  n  y  y  y  y  n  y\n",
       "433  republican  n  n  n  y  y  y  ?  ?  ?  ?  n  y  y  y  n  y\n",
       "434  republican  n  y  n  y  y  y  n  n  n  y  n  y  y  y  ?  n\n",
       "\n",
       "[435 rows x 17 columns]"
      ]
     },
     "execution_count": 744,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "house_df = pd.read_csv(\"../data/house-votes-84.data\", header=None)\n",
    "house_df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 752,
   "metadata": {},
   "outputs": [],
   "source": [
    "def votes(series):\n",
    "    if series == \"y\":\n",
    "        return 1\n",
    "    elif series == \"n\":\n",
    "        return 0\n",
    "    else:\n",
    "        return \"N/A\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 754,
   "metadata": {},
   "outputs": [
    {
     "ename": "ValueError",
     "evalue": "('The truth value of a Series is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all().', 'occurred at index 1')",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mValueError\u001b[0m                                Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-754-1aae6e49c0e6>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[0;32m      1\u001b[0m \u001b[0mhouse_check\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mhouse_df\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0miloc\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;36m1\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;36m16\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m----> 2\u001b[1;33m \u001b[0mhouse_check\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mhouse_check\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mapply\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mvotes\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[1;32m~\\Anaconda3\\lib\\site-packages\\pandas\\core\\frame.py\u001b[0m in \u001b[0;36mapply\u001b[1;34m(self, func, axis, broadcast, raw, reduce, result_type, args, **kwds)\u001b[0m\n\u001b[0;32m   6911\u001b[0m             \u001b[0mkwds\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mkwds\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   6912\u001b[0m         )\n\u001b[1;32m-> 6913\u001b[1;33m         \u001b[1;32mreturn\u001b[0m \u001b[0mop\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mget_result\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m   6914\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   6915\u001b[0m     \u001b[1;32mdef\u001b[0m \u001b[0mapplymap\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mfunc\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\Anaconda3\\lib\\site-packages\\pandas\\core\\apply.py\u001b[0m in \u001b[0;36mget_result\u001b[1;34m(self)\u001b[0m\n\u001b[0;32m    184\u001b[0m             \u001b[1;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mapply_raw\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    185\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 186\u001b[1;33m         \u001b[1;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mapply_standard\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    187\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    188\u001b[0m     \u001b[1;32mdef\u001b[0m \u001b[0mapply_empty_result\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\Anaconda3\\lib\\site-packages\\pandas\\core\\apply.py\u001b[0m in \u001b[0;36mapply_standard\u001b[1;34m(self)\u001b[0m\n\u001b[0;32m    290\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    291\u001b[0m         \u001b[1;31m# compute the result using the series generator\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 292\u001b[1;33m         \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mapply_series_generator\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    293\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    294\u001b[0m         \u001b[1;31m# wrap results\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\Anaconda3\\lib\\site-packages\\pandas\\core\\apply.py\u001b[0m in \u001b[0;36mapply_series_generator\u001b[1;34m(self)\u001b[0m\n\u001b[0;32m    319\u001b[0m             \u001b[1;32mtry\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    320\u001b[0m                 \u001b[1;32mfor\u001b[0m \u001b[0mi\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mv\u001b[0m \u001b[1;32min\u001b[0m \u001b[0menumerate\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mseries_gen\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 321\u001b[1;33m                     \u001b[0mresults\u001b[0m\u001b[1;33m[\u001b[0m\u001b[0mi\u001b[0m\u001b[1;33m]\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mf\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mv\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    322\u001b[0m                     \u001b[0mkeys\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mappend\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mv\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mname\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    323\u001b[0m             \u001b[1;32mexcept\u001b[0m \u001b[0mException\u001b[0m \u001b[1;32mas\u001b[0m \u001b[0me\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m<ipython-input-752-d8af8c1b5224>\u001b[0m in \u001b[0;36mvotes\u001b[1;34m(series)\u001b[0m\n\u001b[0;32m      1\u001b[0m \u001b[1;32mdef\u001b[0m \u001b[0mvotes\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mseries\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m----> 2\u001b[1;33m     \u001b[1;32mif\u001b[0m \u001b[0mseries\u001b[0m \u001b[1;33m==\u001b[0m \u001b[1;34m\"y\"\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m      3\u001b[0m         \u001b[1;32mreturn\u001b[0m \u001b[1;36m1\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      4\u001b[0m     \u001b[1;32melif\u001b[0m \u001b[0mseries\u001b[0m \u001b[1;33m==\u001b[0m \u001b[1;34m\"n\"\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      5\u001b[0m         \u001b[1;32mreturn\u001b[0m \u001b[1;36m0\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\Anaconda3\\lib\\site-packages\\pandas\\core\\generic.py\u001b[0m in \u001b[0;36m__nonzero__\u001b[1;34m(self)\u001b[0m\n\u001b[0;32m   1553\u001b[0m             \u001b[1;34m\"The truth value of a {0} is ambiguous. \"\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1554\u001b[0m             \"Use a.empty, a.bool(), a.item(), a.any() or a.all().\".format(\n\u001b[1;32m-> 1555\u001b[1;33m                 \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m__class__\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m__name__\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m   1556\u001b[0m             )\n\u001b[0;32m   1557\u001b[0m         )\n",
      "\u001b[1;31mValueError\u001b[0m: ('The truth value of a Series is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all().', 'occurred at index 1')"
     ]
    }
   ],
   "source": [
    "house_check = house_df.iloc[:,1:16]\n",
    "house_check = house_check.apply(votes)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 753,
   "metadata": {},
   "outputs": [
    {
     "ename": "ValueError",
     "evalue": "The truth value of a DataFrame is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all().",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mValueError\u001b[0m                                Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-753-70ce0e29ed60>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[1;32m----> 1\u001b[1;33m \u001b[0mvotes\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mhouse_check\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[1;32m<ipython-input-752-d8af8c1b5224>\u001b[0m in \u001b[0;36mvotes\u001b[1;34m(series)\u001b[0m\n\u001b[0;32m      1\u001b[0m \u001b[1;32mdef\u001b[0m \u001b[0mvotes\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mseries\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m----> 2\u001b[1;33m     \u001b[1;32mif\u001b[0m \u001b[0mseries\u001b[0m \u001b[1;33m==\u001b[0m \u001b[1;34m\"y\"\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m      3\u001b[0m         \u001b[1;32mreturn\u001b[0m \u001b[1;36m1\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      4\u001b[0m     \u001b[1;32melif\u001b[0m \u001b[0mseries\u001b[0m \u001b[1;33m==\u001b[0m \u001b[1;34m\"n\"\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      5\u001b[0m         \u001b[1;32mreturn\u001b[0m \u001b[1;36m0\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\Anaconda3\\lib\\site-packages\\pandas\\core\\generic.py\u001b[0m in \u001b[0;36m__nonzero__\u001b[1;34m(self)\u001b[0m\n\u001b[0;32m   1553\u001b[0m             \u001b[1;34m\"The truth value of a {0} is ambiguous. \"\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1554\u001b[0m             \"Use a.empty, a.bool(), a.item(), a.any() or a.all().\".format(\n\u001b[1;32m-> 1555\u001b[1;33m                 \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m__class__\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m__name__\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m   1556\u001b[0m             )\n\u001b[0;32m   1557\u001b[0m         )\n",
      "\u001b[1;31mValueError\u001b[0m: The truth value of a DataFrame is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all()."
     ]
    }
   ],
   "source": [
    "house_check['Age Group'] = resp_df['Age'].apply(age_groups)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 747,
   "metadata": {},
   "outputs": [],
   "source": [
    "x_train, x_test = train_test_split(house_check, test_size=.5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 748,
   "metadata": {},
   "outputs": [
    {
     "ename": "ValueError",
     "evalue": "could not convert string to float: 'n'",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mValueError\u001b[0m                                Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-748-a2e7f1dbe6da>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[1;32m----> 1\u001b[1;33m \u001b[0mmodel\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mfit\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mx_train\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mdrop\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;34m\"party\"\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;36m1\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mx_train\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mparty\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[1;32m~\\Anaconda3\\lib\\site-packages\\sklearn\\tree\\tree.py\u001b[0m in \u001b[0;36mfit\u001b[1;34m(self, X, y, sample_weight, check_input, X_idx_sorted)\u001b[0m\n\u001b[0;32m    814\u001b[0m             \u001b[0msample_weight\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0msample_weight\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    815\u001b[0m             \u001b[0mcheck_input\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mcheck_input\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 816\u001b[1;33m             X_idx_sorted=X_idx_sorted)\n\u001b[0m\u001b[0;32m    817\u001b[0m         \u001b[1;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    818\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\Anaconda3\\lib\\site-packages\\sklearn\\tree\\tree.py\u001b[0m in \u001b[0;36mfit\u001b[1;34m(self, X, y, sample_weight, check_input, X_idx_sorted)\u001b[0m\n\u001b[0;32m    128\u001b[0m         \u001b[0mrandom_state\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mcheck_random_state\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mrandom_state\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    129\u001b[0m         \u001b[1;32mif\u001b[0m \u001b[0mcheck_input\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 130\u001b[1;33m             \u001b[0mX\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mcheck_array\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mX\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mdtype\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mDTYPE\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0maccept_sparse\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;34m\"csc\"\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    131\u001b[0m             \u001b[0my\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mcheck_array\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0my\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mensure_2d\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;32mFalse\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mdtype\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;32mNone\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    132\u001b[0m             \u001b[1;32mif\u001b[0m \u001b[0missparse\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mX\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\Anaconda3\\lib\\site-packages\\sklearn\\utils\\validation.py\u001b[0m in \u001b[0;36mcheck_array\u001b[1;34m(array, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, ensure_min_samples, ensure_min_features, warn_on_dtype, estimator)\u001b[0m\n\u001b[0;32m    494\u001b[0m             \u001b[1;32mtry\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    495\u001b[0m                 \u001b[0mwarnings\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0msimplefilter\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;34m'error'\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mComplexWarning\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 496\u001b[1;33m                 \u001b[0marray\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mnp\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0masarray\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0marray\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mdtype\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mdtype\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0morder\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0morder\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    497\u001b[0m             \u001b[1;32mexcept\u001b[0m \u001b[0mComplexWarning\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    498\u001b[0m                 raise ValueError(\"Complex data not supported\\n\"\n",
      "\u001b[1;32m~\\Anaconda3\\lib\\site-packages\\numpy\\core\\numeric.py\u001b[0m in \u001b[0;36masarray\u001b[1;34m(a, dtype, order)\u001b[0m\n\u001b[0;32m    536\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    537\u001b[0m     \"\"\"\n\u001b[1;32m--> 538\u001b[1;33m     \u001b[1;32mreturn\u001b[0m \u001b[0marray\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0ma\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mdtype\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mcopy\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;32mFalse\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0morder\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0morder\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    539\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    540\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mValueError\u001b[0m: could not convert string to float: 'n'"
     ]
    }
   ],
   "source": [
    "model.fit(x_train.drop(\"party\", 1), x_train.party)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 733,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[(1, 0.0),\n",
       " (2, 0.009347303803909622),\n",
       " (3, 0.08238120383785923),\n",
       " (4, 0.7122313032811503),\n",
       " (5, 0.0),\n",
       " (6, 0.0),\n",
       " (7, 0.019726946633904542),\n",
       " (8, 0.006125063570101961),\n",
       " (9, 0.0),\n",
       " (10, 0.015963634110104864),\n",
       " (11, 0.09890252544369305),\n",
       " (12, 0.016228034895586795),\n",
       " (13, 0.0),\n",
       " (14, 0.0),\n",
       " (15, 0.039093984423689734)]"
      ]
     },
     "execution_count": 733,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "list(zip(x_train.drop(\"party\", axis = 1).columns, model.feature_importances_))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 739,
   "metadata": {},
   "outputs": [],
   "source": [
    "predictions_house = model.predict(x_test.drop(\"party\", 1))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 740,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.9128440366972477"
      ]
     },
     "execution_count": 740,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "accuracy_score(x_test.party, predictions_house)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 741,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[119,  10],\n",
       "       [  9,  80]], dtype=int64)"
      ]
     },
     "execution_count": 741,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "sk.metrics.confusion_matrix(x_test.party, predictions_house)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 742,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "    democrat       0.93      0.92      0.93       129\n",
      "  republican       0.89      0.90      0.89        89\n",
      "\n",
      "    accuracy                           0.91       218\n",
      "   macro avg       0.91      0.91      0.91       218\n",
      "weighted avg       0.91      0.91      0.91       218\n",
      "\n"
     ]
    }
   ],
   "source": [
    "print(classification_report(x_test.party, predictions_house))"
   ]
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
   "version": "3.7.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
