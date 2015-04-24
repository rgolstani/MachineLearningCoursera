# Machine Learning Coursera Project Assignment
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. In this project, the goal is to use data from accelerometers for 6 individuals on four locations:
.the belt   .forearm   .arm    .dumbell 
They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: 
http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset). 
The data for this project come from this source: 
http://groupware.les.inf.puc-rio.br/har. 

## Calling all necessary libraries for the programming 

```r
library(Hmisc)
```

```
## Warning: package 'Hmisc' was built under R version 3.1.2
```

```
## Loading required package: grid
## Loading required package: lattice
## Loading required package: survival
## Loading required package: splines
## Loading required package: Formula
## Loading required package: ggplot2
## 
## Attaching package: 'Hmisc'
## 
## The following objects are masked from 'package:base':
## 
##     format.pval, round.POSIXt, trunc.POSIXt, units
```

```r
library(caret)
```

```
## Warning: package 'caret' was built under R version 3.1.2
```

```
## 
## Attaching package: 'caret'
## 
## The following object is masked from 'package:survival':
## 
##     cluster
```

```r
library(randomForest)
```

```
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
## 
## Attaching package: 'randomForest'
## 
## The following object is masked from 'package:Hmisc':
## 
##     combine
```

```r
library(foreach)
library(doParallel)
```

## Make sure we are at the working directory for this project

```r
setwd("~/Desktop/MachineLearningCoursera")
```
## Get the data from online sources provided by course project


```r
if(!file.exists("traindata.csv")) {
  file <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
  download.file(file, destfile = "traindata.csv", method = "curl")
}

if(!file.exists("testdata.csv")) {
  file <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
  download.file(file, destfile = "testdata.csv", method = "curl")
}
```

## Cross Validation

Random Forest method has its own internal cross-validation algorithms which protect against overfitting.




Looking the first glance at the data shows that there are a lot of fields as "NA" or " " and some as "#DIV/0!". So to clean teh data, it's advised to get ride of columns with too many blanks or NA or #DIV/0!

## Clean the Data

We replace the " ","DIV/0!", "NA" all to NA


```r
train<- read.csv("traindata.csv", na.strings=c("NA","#DIV/0!",""))
test<- read.csv("testdata.csv", na.strings=c("NA","#DIV/0!",""))
```

We get ride of columns that have all NAs. To do this we get ride of columns that end up with the sum of 0.



```r
train.na.removed <- train[ ,(colSums(is.na(train)) == 0)]
test.na.removed <- test[ ,(colSums(is.na(test)) == 0)]
```

We also notice that the first 7 columns are not serving us in modeling and they just slow down the future model fitting.


```r
cleaned.train <- train.na.removed[ , c(7:60)]
cleaned.test <- test.na.removed[ , c(7:60)]
```

## Partitioning train data 
We dedicate %60 of our train data set to model fitting and %40 for testing. Using the cleaned set of data.


```r
idx <- createDataPartition(y=cleaned.train$classe, p=0.60, list=FALSE )
training.set <- cleaned.train[idx,]
testing.set <- cleaned.train[-idx,]
```

## Using parallel processing to speed up random forest model.


```r
registerDoParallel()
x <- training.set[-ncol(training.set)]
y <- training.set$classe

rf <- foreach(ntree=rep(150, 6), .combine=randomForest::combine, .packages='randomForest') %dopar% {
  randomForest(x, y, ntree=ntree) 
}
```

## In Sample Accuracy in the training set 


```r
predictions1 <- predict(rf, newdata=training.set)
confusionMatrix(predictions1,training.set$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 3348    0    0    0    0
##          B    0 2279    0    0    0
##          C    0    0 2054    0    0
##          D    0    0    0 1930    0
##          E    0    0    0    0 2165
## 
## Overall Statistics
##                                      
##                Accuracy : 1          
##                  95% CI : (0.9997, 1)
##     No Information Rate : 0.2843     
##     P-Value [Acc > NIR] : < 2.2e-16  
##                                      
##                   Kappa : 1          
##  Mcnemar's Test P-Value : NA         
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
## Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
## Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Prevalence             0.2843   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2843   0.1935   0.1744   0.1639   0.1838
## Detection Prevalence   0.2843   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000
```

We find %100 accuracy on our training set

## Out of Sample Accuracy in testing set of training data


```r
predictions2 <- predict(rf, newdata=testing.set)
confusionMatrix(predictions2,testing.set$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2231    4    0    0    0
##          B    0 1514   11    0    0
##          C    0    0 1357    8    0
##          D    0    0    0 1278    5
##          E    1    0    0    0 1437
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9963          
##                  95% CI : (0.9947, 0.9975)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9953          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9996   0.9974   0.9920   0.9938   0.9965
## Specificity            0.9993   0.9983   0.9988   0.9992   0.9998
## Pos Pred Value         0.9982   0.9928   0.9941   0.9961   0.9993
## Neg Pred Value         0.9998   0.9994   0.9983   0.9988   0.9992
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2843   0.1930   0.1730   0.1629   0.1832
## Detection Prevalence   0.2849   0.1944   0.1740   0.1635   0.1833
## Balanced Accuracy      0.9994   0.9978   0.9954   0.9965   0.9982
```

WooHoo the accuracy is pretty good 0.9985. So this Model is capable of forcasting our future data set.

## Future Prediction for the real test data


```r
predict.the.future <- predict(rf, newdata=cleaned.test)
predict.the.future
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```

