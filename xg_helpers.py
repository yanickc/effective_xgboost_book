import subprocess
import urllib.request
import zipfile

import pandas as pd
import xgboost as xgb
from feature_engine import encoding, imputation
from sklearn import base, pipeline


def extract_zip(src, dst, member_name):
    """Extract a member file from a zip file and read it into a pandas
    DataFrame.

    Parameters:
        src (str): URL of the zip file to be downloaded and extracted.
        dst (str): Local file path where the zip file will be written.
        member_name (str): Name of the member file inside the zip file
            to be read into a DataFrame.

    Returns:
        pandas.DataFrame: DataFrame containing the contents of the
            member file.
    """
    url = src
    fname = dst
    fin = urllib.request.urlopen(url)
    data = fin.read()
    with open(dst, mode="wb") as fout:
        fout.write(data)
    with zipfile.ZipFile(dst) as z:
        kag = pd.read_csv(z.open(member_name))
        kag_questions = kag.iloc[0]
        raw = kag.iloc[1:]
        return raw


def get_rawX_y(df, y_col):
    raw = df.query(
        'Q3.isin(["United States of America", "China", "India"]) '
        'and Q6.isin(["Data Scientist", "Software Engineer"])'
    )
    return raw.drop(columns=[y_col]), raw[y_col]


def tweak_kag(df_: pd.DataFrame) -> pd.DataFrame:
    """
    Tweak the Kaggle survey data and return a new DataFrame.

    This function takes a Pandas DataFrame containing Kaggle
    survey data as input and returns a new DataFrame. The
    modifications include extracting and transforming certain
    columns, renaming columns, and selecting a subset of columns.

    Parameters
    ----------
    df_ : pd.DataFrame
        The input DataFrame containing Kaggle survey data.

    Returns
    -------
    pd.DataFrame
        The new DataFrame with the modified and selected columns.
    """
    return (
        df_.assign(
            age=df_.Q2.str.slice(0, 2).astype(int),
            education=df_.Q4.replace(
                {
                    "Master’s degree": 18,
                    "Bachelor’s degree": 16,
                    "Doctoral degree": 20,
                    "Some college/university study without earning a bachelor’s degree": 13,
                    "Professional degree": 19,
                    "I prefer not to answer": None,
                    "No formal education past high school": 12,
                }
            ),
            major=(
                df_.Q5.pipe(topn, n=3).replace(
                    {
                        "Computer science (software engineering, etc.)": "cs",
                        "Engineering (non-computer focused)": "eng",
                        "Mathematics or statistics": "stat",
                    }
                )
            ),
            years_exp=(
                df_.Q8.str.replace("+", "", regex=False)
                .str.split("-", expand=True)
                .iloc[:, 0]
                .astype(float)
            ),
            compensation=(
                df_.Q9.str.replace("+", "", regex=False)
                .str.replace(",", "", regex=False)
                .str.replace("500000", "500", regex=False)
                .str.replace(
                    "I do not wish to disclose my approximate yearly compensation",
                    "0",
                    regex=False,
                )
                .str.split("-", expand=True)
                .iloc[:, 0]
                .fillna(0)
                .astype(int)
                .mul(1_000)
            ),
            python=df_.Q16_Part_1.fillna(0).replace("Python", 1),
            r=df_.Q16_Part_2.fillna(0).replace("R", 1),
            sql=df_.Q16_Part_3.fillna(0).replace("SQL", 1),
        )  # assign
        .rename(columns=lambda col: col.replace(" ", "_"))
        .loc[
            :,
            "Q1,Q3,age,education,major,years_exp,compensation,"
            "python,r,sql".split(","),
        ]
    )


def topn(ser, n=5, default="other"):
    """
    Replace all values in a Pandas Series that are not among
    the top `n` most frequent values with a default value.

    This function takes a Pandas Series and returns a new
    Series with the values replaced as described above. The
    top `n` most frequent values are determined using the
    `value_counts` method of the input Series.

    Parameters
    ----------
    ser : pd.Series
        The input Series.
    n : int, optional
        The number of most frequent values to keep. The
        default value is 5.
    default : str, optional
        The default value to use for values that are not among
        the top `n` most frequent values. The default value is
        'other'.

    Returns
    -------
    pd.Series
        The modified Series with the values replaced.
    """
    counts = ser.value_counts()
    return ser.where(ser.isin(counts.index[:n]), default)


class TweakKagTransformer(base.BaseEstimator, base.TransformerMixin):
    """
    A transformer for tweaking Kaggle survey data.

    This transformer takes a Pandas DataFrame containing
    Kaggle survey data as input and returns a new version of
    the DataFrame. The modifications include extracting and
    transforming certain columns, renaming columns, and
    selecting a subset of columns.

    Parameters
    ----------
    ycol : str, optional
        The name of the column to be used as the target variable.
        If not specified, the target variable will not be set.

    Attributes
    ----------
    ycol : str
        The name of the column to be used as the target variable.
    """

    def __init__(self, ycol=None):
        self.ycol = ycol

    def transform(self, X):
        return tweak_kag(X)

    def fit(self, X, y=None):
        return self


kag_pl = pipeline.Pipeline(
    [
        ("tweak", TweakKagTransformer()),
        (
            "cat",
            encoding.OneHotEncoder(
                top_categories=5, drop_last=True, variables=["Q1", "Q3", "major"]
            ),
        ),
        (
            "num_impute",
            imputation.MeanMedianImputer(
                imputation_method="median", variables=["education", "years_exp"]
            ),
        ),
    ]
)


def my_dot_export(xg, num_trees, filename, title="", direction="TB"):
    """Exports a specified number of trees from an XGBoost model as a graph
    visualization in dot and png formats.

    Args:
        xg: An XGBoost model.
        num_trees: The number of tree to export.
        filename: The name of the file to save the exported visualization.
        title: The title to display on the graph visualization (optional).
        direction: The direction to lay out the graph, either 'TB' (top to
            bottom) or 'LR' (left to right) (optional).
    """
    res = xgb.to_graphviz(xg, num_trees=num_trees)
    content = f"""    node [fontname = "Roboto Condensed"];
    edge [fontname = "Roboto Thin"];
    label = "{title}"
    fontname = "Roboto Condensed"
    """
    out = res.source.replace(
        "graph [ rankdir=TB ]", f"graph [ rankdir={direction} ];\n {content}"
    )
    # dot -Gdpi=300 -Tpng -ocourseflow.png courseflow.dot
    dot_filename = filename
    with open(dot_filename, "w") as fout:
        fout.write(out)
    png_filename = dot_filename.replace(".dot", ".png")
    subprocess.run(f"dot -Gdpi=300 -Tpng -o{png_filename} {dot_filename}".split())
