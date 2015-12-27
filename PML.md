##**Practical Machine Learning Project**
_Mader_

_December,2015_



###**Background Introduction**

These are files from project assignment of Coursera's MOOC Practical Machine Learning from Johns Hopkins University. 

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. 

One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. 

The goal of this project is to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. 

More information is available from the website here:(http://groupware.les.inf.puc-rio.br/har)(see the section on the Weight Lifting Exercise Dataset). 


###**Data**##

The training data for this project are available here: 

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here: 

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

The data for this project come from this source: http://groupware.les.inf.puc-rio.br/har

First, we download the data from the source

```r
TrainURL<-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
TestURL<-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
```

Then, the data uploaded into R(using Rstudio) and replacing the NA, #DIV/0! and empty fields as NA:

```r
training <- read.csv(url(TrainURL), na.strings=c("NA","#DIV/0!",""))
testing <- read.csv(url(TestURL), na.strings=c("NA","#DIV/0!",""))
```

Take a look at the content o the training dataset, and the _classe_ variable which we need to predict with

```r
str(training, list.len=10)
```

```
## 'data.frame':	19622 obs. of  160 variables:
##  $ X                       : int  1 2 3 4 5 6 7 8 9 10 ...
##  $ user_name               : Factor w/ 6 levels "adelmo","carlitos",..: 2 2 2 2 2 2 2 2 2 2 ...
##  $ raw_timestamp_part_1    : int  1323084231 1323084231 1323084231 1323084232 1323084232 1323084232 1323084232 1323084232 1323084232 1323084232 ...
##  $ raw_timestamp_part_2    : int  788290 808298 820366 120339 196328 304277 368296 440390 484323 484434 ...
##  $ cvtd_timestamp          : Factor w/ 20 levels "02/12/2011 13:32",..: 9 9 9 9 9 9 9 9 9 9 ...
##  $ new_window              : Factor w/ 2 levels "no","yes": 1 1 1 1 1 1 1 1 1 1 ...
##  $ num_window              : int  11 11 11 12 12 12 12 12 12 12 ...
##  $ roll_belt               : num  1.41 1.41 1.42 1.48 1.48 1.45 1.42 1.42 1.43 1.45 ...
##  $ pitch_belt              : num  8.07 8.07 8.07 8.05 8.07 8.06 8.09 8.13 8.16 8.17 ...
##  $ yaw_belt                : num  -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 ...
##   [list output truncated]
```

```r
table(training$classe)
```

```
## 
##    A    B    C    D    E 
## 5580 3797 3422 3216 3607
```
Using prop.table to express Table Entries proportions with respect to the different marginal distributions

```r
prop.table(table(training$user_name, training$classe), 1)
```

```
##           
##                    A         B         C         D         E
##   adelmo   0.2993320 0.1993834 0.1927030 0.1323227 0.1762590
##   carlitos 0.2679949 0.2217224 0.1584190 0.1561697 0.1956941
##   charles  0.2542421 0.2106900 0.1524321 0.1815611 0.2010747
##   eurico   0.2817590 0.1928339 0.1592834 0.1895765 0.1765472
##   jeremy   0.3459730 0.1437390 0.1916520 0.1534392 0.1651969
##   pedro    0.2452107 0.1934866 0.1911877 0.1796935 0.1904215
```

```r
prop.table(table(training$classe))
```

```
## 
##         A         B         C         D         E 
## 0.2843747 0.1935073 0.1743961 0.1638977 0.1838243
```
###**Data Cleaning**##
Using  basic data clean-up, we do some data clean-up by removing columns 1 to 6, which are there just for information and reference purposes:

```r
training <- training[, 7:160]
testing  <- testing[, 7:160]
```
and removing all columns that are mostly NA:

```r
data <- apply(!is.na(training), 2, sum) > 19621  # which is the number of observations
training <- training[, data]
testing  <- testing[, data]
```


###**Data Splitting**##

Using Caret packages, the dataset split into two for **cross validation** purposes. We allocated 70% for training & the remainder 30% will be used for test and validation


```r
library(caret)

set.seed(333)
inTrain <- createDataPartition(y=training$classe, p=0.70, list=FALSE)
train1  <- training[inTrain,]
test1  <- training[-inTrain,]
dim(train1)
```

```
## [1] 13737    54
```

```r
dim(test1)
```

```
## [1] 5885   54
```
##**Prediction: Decision Tree**##
Using ML algorithms method for prediction

```r
library(rpart)
modFit1 <- rpart(classe ~ ., data=train1, method="class")
library(rattle)
fancyRpartPlot(modFit1)
```

![](Figs/unnamed-chunk-9-1.png) 

```r
predictions1 <- predict(modFit1, test1, type = "class")
confusionMatrix(predictions1, test1$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1496  240   43  104   64
##          B   40  650   70   33  112
##          C   21   69  818  150   86
##          D   97  145   64  638  137
##          E   20   35   31   39  683
## 
## Overall Statistics
##                                           
##                Accuracy : 0.7281          
##                  95% CI : (0.7166, 0.7395)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.6545          
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.8937   0.5707   0.7973   0.6618   0.6312
## Specificity            0.8929   0.9463   0.9329   0.9100   0.9740
## Pos Pred Value         0.7684   0.7182   0.7150   0.5902   0.8453
## Neg Pred Value         0.9548   0.9018   0.9561   0.9321   0.9214
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2542   0.1105   0.1390   0.1084   0.1161
## Detection Prevalence   0.3308   0.1538   0.1944   0.1837   0.1373
## Balanced Accuracy      0.8933   0.7585   0.8651   0.7859   0.8026
```
##**Prediction: Random Forest**##
Using Random Forest method for prediction

```r
library(randomForest)
modFit2 <- randomForest(classe ~. , data=train1)
predictions2 <- predict(modFit2, test1, type = "class")
confusionMatrix(predictions2, test1$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1674    1    0    0    0
##          B    0 1138    3    0    0
##          C    0    0 1023    8    0
##          D    0    0    0  956    5
##          E    0    0    0    0 1077
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9971          
##                  95% CI : (0.9954, 0.9983)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9963          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9991   0.9971   0.9917   0.9954
## Specificity            0.9998   0.9994   0.9984   0.9990   1.0000
## Pos Pred Value         0.9994   0.9974   0.9922   0.9948   1.0000
## Neg Pred Value         1.0000   0.9998   0.9994   0.9984   0.9990
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2845   0.1934   0.1738   0.1624   0.1830
## Detection Prevalence   0.2846   0.1939   0.1752   0.1633   0.1830
## Balanced Accuracy      0.9999   0.9992   0.9977   0.9953   0.9977
```

##**Project Submission**##
Base on prediction method use in previous section, Random Forests resulting with better accuracy. Therefore, we use Random Forests with the following formula, which yielded a much better prediction in in-sample:

```r
prediction<- predict(modFit2, testing, type = "class")
```
Function to generate files with predictions to submit for assignment

```r
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(prediction)
```

