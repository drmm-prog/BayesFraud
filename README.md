# Bayesian Fraud Engine

This code makes an estimate of the probability of a transaction being a fraud given some aggregated historic transaction data (given in a file), a set of fraud detection rules (also in the file), and a set of fraud detection rules triggered by the transaction at hand. 

The theory behind the calculation is given in `docs/Fraud Detection Theory.pdf`. It is a simple model assuming that transactions are independent of one another and that rules are triggered independently.  

The `Bayesian_Fraud_Engine` takes the aggregated data and rules upon initialisation, and then subsequently takes individual sets of triggered rules for each each transaction. 

## Running the code

The code requires Python 3 to work (it is not compatible with Python 2 because of the differences in integer division). You will also need installed the packages `numpy`, `csv`, and `functools`. 

Running the test program can be done by calling the script with one argument, which is the location of the data file:
```
python3 run_example.py test_data/testData.csv
```

The data is a `.csv` file with the following format:
```
totals;<num_transactions>;<num_frauds>
rule_1;<num_frauds_r>;<num_genuine_r>
...
```
The first line must start with `totals`, followed by the total number of transactions, then the total number of frauds. The following lines each start with a unique rule name, followed by first the number of times the rule has been triggered by a fraudulent transaction, then the number of times the rule has been triggered by a genuine transaction. 

To run it in your own code simply import the package, then create an engine object, initialised with your data file:
```
from Fraud_Detection import Bayesian_Fraud_Engine as BFE
engine = BFE.Bayesian_Fraud_Engine(data_file_path)
```
you can then calculate the posterior using:
```
engine.posterior(S)
```
where S _must_ be a Python `set`. (You may convert lists to sets using `set(L)`.)