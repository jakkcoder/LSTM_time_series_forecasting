import numpy as np 
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion, Pipeline 
from scipy.stats import skew ,kurtosis,zscore,chisquare
import scipy.stats as stats
from IPython.display import display

class EDA(BaseEstimator, TransformerMixin):
    
    
    
    """
    Developer: jaybshankar@yahoo.com
    
    This class will be used to perform EDA on a given dataset, and calculate feature importance using
    One-way ANOVA : For categorical target and numerical feature, remember A large F ratio means that the 
    variation among groups means more than you'd expect to see by chance.
    
    
    
    Chi-Square Test: For categorical targets and categorical features, In feature selection, 
    we aim to select the features which are highly dependent on the response. So high Chi-Square value 
    indicates that the hypothesis of independence is incorrect. In short as the higher value of chi-square 
    represents a better feature.
    
    
    Correlation Test: This test is used to find the correlation between one hot encoded vector and other 
    numerical features, a higher value of correlation means better the attribute. 
    
    How to use the function:
    
    
    Initiate the object with the EDA class
    
    Ex. 
    
    
    >> obj_eda = EDA(target_col = 'race')
    >> obj_eda.fit_transform(df)
    
    
    Output: 
    
    You can access all the eda using the below methods
    obj_eda.summary : Higher level summary of dataset
    obj_eda.corr_df : Correaltion with target
    obj_eda.univariate_numerical_results : univariate numerical features summary 
    obj_eda.univariate_categorical_df :univariate Categorical features summary 
    obj_eda.anova_scores_df = ANOVA results for feature selection
    obj_eda.Chi_scores_df =  Chi Square scores for feature selection
    
    
    
    
    Parameters
    ----------
    target_col: str 
        Name of the target feature
    """ 
    
    
    def __init__(self,target_col = None):
        self.target_col = target_col
        self.univariate_numerical_results  = {}
        self.variable_types_results = {}
        self.univariate_categorical_results = {}
        self.biveriate_categorical_results = {}
        self.biveriate_anova_results = {}
        self.biveriate_chi_results = {}
        

        
    def fit(self, X, y=None):
        self.df = X
        self._run_EDA = self._dataset_statistics()
        self._run_EDA2 = self._variable_types()
        return self
        
    
    def transform(self,X):
        print('------------------------------------------------------------------------------------\n')
        for cols in self.df.columns:
            if (self.df[cols].dtype == 'O') |(self.df[cols].dtype == str):
                if self.df[cols].is_unique:
                    print(f"{cols} is having all distinct values")    
                else:
                    self._univariate_categorical(column_name = cols)
                    print(f"univeriate categorical analysis completed for column: {cols}")
                    if self.target_col:
                        self._biveriate_categorical(cols,self.target_col)
            
                
            ### if current column is numerical
            elif (self.df[cols].dtype == float) |(self.df[cols].dtype == int):
                self._univariate_numerical(column_name=cols)
                print(f"univeriate numerical analysis completed for column: {cols}")
                if self.target_col:
                    self._biveriate_categorical(cols,self.target_col)
        
        if self.target_col:         
            self._correlation_testing()
            self._conv_results_to_df()
        
    def _dataset_statistics(self):
        print ('\033[1m' + 'Calculating Dataset Statistics' +'\033[0m')
        self.missing_count = self.df.isna().sum().sum()
        self.row_count= self.df.shape[0]
        self.column_count = self.df.shape[1]
        
        self.all_nan_cols = self.df.isna().all()[self.df.isna().all()].index.tolist()
        ## droping all null value columns for further analysis
        self.df = self.df.drop(columns = self.all_nan_cols)
    
    def _variable_types(self):
        self.Categorical_count = ((self.df.dtypes == 'object')|(self.df.dtypes == 'str')).sum()
        self.Numeric_count = ((self.df.dtypes == 'int')|(self.df.dtypes == 'float')).sum()
        self.Unknown_count = self.column_count -self.Categorical_count - self.Numeric_count
        
        
    def _univariate_numerical(self,column_name,threshold = 3):
        results= {}
        '''function will be used for univeriate analysis on the numerical columns
        
        Parameters:
        column_name - columns name for univeriate EDA
        threshold - Ouliter detecttion threshold Ex 3 is 3 sigma limit
        
        ''' 
        
        
        results['missing_percentage']=(self.df[column_name].isna().sum()/self.row_count)*100
        results['mean_value']= self.df[column_name].mean()
        results['standard_deviation']= self.df[column_name].std()
        
        skew_c =skew(self.df[column_name].dropna())
        results['skew'] = skew_c
        if skew_c>1 : results['skewness']="Left skewed"
        elif skew_c<0:results['skewness'] ="Normally distributed"
        else: results['skewness'] = "Right skewed"
        ### Kertosis 
        kurtosis_c =kurtosis(self.df[column_name].dropna())
        results['kurtosis'] = kurtosis_c
        
        
        if kurtosis_c>1 : results['kurtosis_R'] = "higher kurtosis, heavier tail"
        elif kurtosis_c<0: results['kurtosis_R'] = "low kurtosis,thinner tail"
        else: results['kurtosis_R']  = "Normally distributed"
        
        
        

        ## Testing for outliers
        zscore_c = zscore(self.df[column_name].dropna())
        right_side_outliers  = self.df[column_name].loc[zscore_c[zscore_c>threshold].index]
        left_side_outliers  = self.df[column_name].loc[zscore_c[zscore_c<-threshold].index]
        total_outliers = len(right_side_outliers) + len(left_side_outliers)
        if total_outliers>0:
            if len(self.df[column_name].unique())>2:
                ### replcing outliers with upper and lower limits
                upper_lim = (self.df[column_name].std()*4) +self.df[column_name].mean()
                lower_lim = -(self.df[column_name].std()*4) + self.df[column_name].mean()
                self.df[column_name].loc[right_side_outliers.index] = upper_lim 
                self.df[column_name].loc[left_side_outliers.index] = lower_lim

        ### Normality testing using Chisquare test
        
        stat,p = chisquare(self.df[column_name].dropna())
        results['chi_stat_normality'] = [stat,p]
        self.univariate_numerical_results[column_name] = results
        
    def _univariate_categorical(self,column_name):
        results = {}
        results['missing_percentage'] = (self.df[column_name].isna().sum()/self.row_count)*100
        results['distinct_values'] = len(self.df[column_name].unique())
        results['unique_percentage'] = round((results['distinct_values']/self.row_count)*100,2)
        self.univariate_categorical_results[column_name] = results
        
        
    def _biveriate_categorical(self,column_name,target_col):
        if (self.df[target_col].dtype == "O") | (self.df[target_col].dtype == "str"):
            ## if current column is numerical
            if (self.df[target_col].dtype == "int") | (self.df[column_name].dtype == float):
                results_cat_num = {}

                ### One Way ANOVA tests the relationship between categorical predictor vs continuous response.
                fvalue, pvalue = stats.f_oneway(*[group[1][column_name].dropna().values 
                                                  for group in self.df[[column_name,target_col]].groupby(target_col)])
                results_cat_num['ANOVA_results'] = [fvalue, pvalue]
                self.biveriate_categorical_results[column_name] = results_cat_num
                self.biveriate_anova_results[column_name] = fvalue

            ### if current variable is categorical
            else:
                results_cat_cat = {}
                ### Performing chi square test for independence
                chi2_stat, p, dof, expected = stats.chi2_contingency(pd.crosstab(self.df[column_name],self.df[target_col]))
                results_cat_cat['chi2_stat'] = [chi2_stat,p]
                self.biveriate_chi_results[column_name] = chi2_stat 
            
                
                
    def _correlation_testing(self):
        """This function calculates correlation between all numerical 
        columns and one hot encoded target vector"""
        target_cols = self.df['race'].unique()
        feature_cols = [col for col in self.df.columns if not col in target_cols ]

        corr_d = {}
        for cols in feature_cols:
            if not self.df[cols].dtype == 'O':
                small_df = self.df[np.hstack([target_cols,cols])].dropna()
                corr_d[cols] = small_df[target_cols].corrwith(small_df[cols]).to_dict()
                
        if len(corr_d)>0:
            self.corr_df =pd.DataFrame(corr_d).T  
            
        
        
         
            
    def summary(self):
        print ('\033[1m' + 'Dataset statistics' +'\033[0m',
        "\n","Number of variables:           ",self.column_count,"\n",
        "Number of observations:        ",self.row_count,"\n",
        "Missing cells:                 ",self.missing_count ,"\n",
        "Missing cells (%):             ",round((self.missing_count/(self.row_count*self.column_count))*100,2) ,"\n",
        "Empty variables count:         ", len(self.all_nan_cols) ,"\n",
        "Empty variables/all nulls:     ", self.all_nan_cols ,"\n",
        '------------------------------------------------------------------------------------\n',
        '\033[1m'+ "Variable types"+'\033[0m',
         "\n","Categorical variables:          ",self.Categorical_count,"\n",
         "Numerical variables:            ",self.Numeric_count,"\n",
         "Unknown variables:              ",self.Unknown_count,"\n",
        '------------------------------------------------------------------------------------\n\n',
        '\033[1m'+ "Univeriate EDA of Numerical Features"+'\033[0m\n')
        if len(self.univariate_numerical_results)>0:
            
            self.univariate_numerical_df = pd.DataFrame(self.univariate_numerical_results).T
            display(self.univariate_numerical_df)
        else:
            print("No Numerical variable Found in the DataFrame/if exist please fix the data types")
        
        print('\033[1m'+ "Univeriate EDA of Categorical Features"+'\033[0m\n')
        if len(self.univariate_categorical_results)>0:
            self.univariate_categorical_df = pd.DataFrame(self.univariate_categorical_results).T
            display(self.univariate_categorical_df)
        else:
            print("No Categorical variable Found in the DataFrame/if exist please fix the data types")
         
        
        print('\033[1m'+ "Bivariate EDA of Categorical Features"+'\033[0m\n',
              '\033[1m'+ "Numerical Features ANOVA Scores"+'\033[0m\n')
        if len(self.biveriate_anova_results)>0:
            self.anova_scores_df = pd.DataFrame(self.biveriate_anova_results,index = ['Anova F ratio']).T.sort_values(by ='Anova F ratio',ascending= False)
            display(self.anova_scores_df)
        else:
            print("No Target columns Provided")
            
            
       
        print('\n \033[1m'+ "Categorical Features Chi Square Scores"+'\033[0m\n')
        
        if len(self.biveriate_chi_results)>0:
            self.Chi_scores_df = pd.DataFrame(self.biveriate_chi_results,index=['chi-squared scores']).T.sort_values(by ='chi-squared scores',ascending= 0) 
            display(self.Chi_scores_df)
        else:
            print("No Target columns Provided")
        
        print('\n \033[1m'+ "Correltion with target"+'\033[0m\n')
        if 'self.corr_df' in locals():
            display(self.corr_df)
        else:
            print("No Target columns Provided")