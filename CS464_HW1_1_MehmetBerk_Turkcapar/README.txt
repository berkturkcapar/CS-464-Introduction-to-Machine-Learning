To run the code numpy and pandas libraries are required. 

There are three funuctions:
- multinomialNaiveBayes
It takes the array of data as inputs. 
It also takes an input called 'alpha' for Dirichlet Prior. 
If alpha is not provided its default value is 0
- bernoulliNaiveBayes
It takes the array of data as inputs. 
- createCSVFile
takes the confusion matrix as input
It also takes an input called 'alpha' for naming of files

To use the code:
python3 q2main.py

The code will output three CSV files:
- mnb0: confusion matrix of Multinomial Naive Bayes (without Dirichlet)
- mnb5: confusion matrix of Multinomial Naive Bayes (with Dirichlet and alpha = 5)
- bnb: confusion matrix of Bernoulli Naive Bayes