Most of these scripts do not run without accessing the correct directories. The scripts in this folder are organized in the following way:
- `save_*.py` files are used to save dataframes from Reddit and GDelt data. They are based on PySpark. 
- `explore_*.py` files are about data exploration, logistic regression and survival analysis.
- `eng_*.py` files are related to feature engeneering, they are used to transform collected data in order to apply causal analysis.
- `causal_*.py` files are used for causal inference. The methods and results in the paper are based on these scripts.