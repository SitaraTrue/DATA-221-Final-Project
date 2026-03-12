from sklearn.impute import KNNImputer
import pandas as pd

diabetes_df = pd.read_csv("diabetes.csv")

num_neighbors = 5
imputer = KNNImputer(n_neighbors=num_neighbors)
cleaned_diabetes_df = diabetes_df.drop(["Pregnancies", "Outcome"], axis=1)

imputed_diabetes_df = pd.DataFrame(imputer.fit_transform(cleaned_diabetes_df), columns=cleaned_diabetes_df.columns)

