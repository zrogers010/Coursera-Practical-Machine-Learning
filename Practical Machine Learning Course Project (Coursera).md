Practical Machine Learning Courser Project (Coursera)
========================================================

## About the Data
The data for this assignment is from 6 participants who wore accelerometers on the belt, forearm, arm, and dumbbells during exercise routines. The data are split into a training group (19,622) observations and testing group (20 observations). Participants in this study were asked to do a "Dumbbell Biceps Curl" in five different ways, including using correct form and four common mistakes.

### Goals

1. Predict the manner in which the participants did the exercise indicated by the 'classe' variable.
2. Build a prediction model, in this case using the Random Forest model.
3. Calculate the in sample accuracy and out of sample error.
4. Use the prediction model to predict on the 20 different test cases provided.


### Getting the Data
First, install necessary packages and check my working directory

```r
library(caret)
```

```
## Warning: package 'caret' was built under R version 3.1.2
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
library(dplyr)
```

```
## 
## Attaching package: 'dplyr'
## 
## The following objects are masked from 'package:stats':
## 
##     filter, lag
## 
## The following objects are masked from 'package:base':
## 
##     intersect, setdiff, setequal, union
```

```r
library(randomForest)
```

```
## Warning: package 'randomForest' was built under R version 3.1.2
```

```
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

```r
#getwd()
```

Download the Training and Testing Data sets, then read them into r while assigning NA values to "NA", "#DIV/0!", and "" values.


```r
training_data_url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
test_data_url <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

training_data <- read.csv("pml-training.csv", na.strings = c("NA", "#DIV/0!", ""))
test_data <- read.csv("pml-testing.csv", na.strings = c("NA", "#DIV/0!", ""))
```

### Pre-process the Data
Split the training data for 60% training and 40% testing


```r
training <- createDataPartition(y = training_data$classe, p=0.6, list=FALSE)
train_set <- training_data[training, ]
test_set <- training_data[-training, ]
```

### Clean the data by determining columns with many NA values and removing them from the data set.  Also remove columns that are not exercise movement related.

```r
# Remove Columns with lots of NA Values
sum(is.na(train_set))
```

```
## [1] 1156629
```

```r
nas <- sapply(train_set, function(x) { sum(is.na(x))})
table(nas)
```

```
## nas
##     0 11547 11548 11549 11550 11551 11555 11567 11587 11588 11590 11591 
##    60    68     1     1     5     4     2     2     1     6     1     1 
## 11593 11776 
##     2     6
```

```r
# We can see that 60 columns dont have any NAs, while the others contain many (over 11,000 for the train_set). Lets remove those.
remove_cols <- names(nas[nas > 0])
train_set <- train_set[, !names(train_set) %in% remove_cols]
str(train_set)
```

```
## 'data.frame':	11776 obs. of  60 variables:
##  $ X                   : int  1 2 3 5 6 8 9 11 12 16 ...
##  $ user_name           : Factor w/ 6 levels "adelmo","carlitos",..: 2 2 2 2 2 2 2 2 2 2 ...
##  $ raw_timestamp_part_1: int  1323084231 1323084231 1323084231 1323084232 1323084232 1323084232 1323084232 1323084232 1323084232 1323084232 ...
##  $ raw_timestamp_part_2: int  788290 808298 820366 196328 304277 440390 484323 500302 528316 644302 ...
##  $ cvtd_timestamp      : Factor w/ 20 levels "02/12/2011 13:32",..: 9 9 9 9 9 9 9 9 9 9 ...
##  $ new_window          : Factor w/ 2 levels "no","yes": 1 1 1 1 1 1 1 1 1 1 ...
##  $ num_window          : int  11 11 11 12 12 12 12 12 12 12 ...
##  $ roll_belt           : num  1.41 1.41 1.42 1.48 1.45 1.42 1.43 1.45 1.43 1.48 ...
##  $ pitch_belt          : num  8.07 8.07 8.07 8.07 8.06 8.13 8.16 8.18 8.18 8.15 ...
##  $ yaw_belt            : num  -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 ...
##  $ total_accel_belt    : int  3 3 3 3 3 3 3 3 3 3 ...
##  $ gyros_belt_x        : num  0 0.02 0 0.02 0.02 0.02 0.02 0.03 0.02 0 ...
##  $ gyros_belt_y        : num  0 0 0 0.02 0 0 0 0 0 0 ...
##  $ gyros_belt_z        : num  -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 0 ...
##  $ accel_belt_x        : int  -21 -22 -20 -21 -21 -22 -20 -21 -22 -21 ...
##  $ accel_belt_y        : int  4 4 5 2 4 4 2 2 2 4 ...
##  $ accel_belt_z        : int  22 22 23 24 21 21 24 23 23 23 ...
##  $ magnet_belt_x       : int  -3 -7 -2 -6 0 -2 1 -5 -2 0 ...
##  $ magnet_belt_y       : int  599 608 600 600 603 603 602 596 602 592 ...
##  $ magnet_belt_z       : int  -313 -311 -305 -302 -312 -313 -312 -317 -319 -305 ...
##  $ roll_arm            : num  -128 -128 -128 -128 -128 -128 -128 -128 -128 -129 ...
##  $ pitch_arm           : num  22.5 22.5 22.5 22.1 22 21.8 21.7 21.5 21.5 21.3 ...
##  $ yaw_arm             : num  -161 -161 -161 -161 -161 -161 -161 -161 -161 -161 ...
##  $ total_accel_arm     : int  34 34 34 34 34 34 34 34 34 34 ...
##  $ gyros_arm_x         : num  0 0.02 0.02 0 0.02 0.02 0.02 0.02 0.02 0.02 ...
##  $ gyros_arm_y         : num  0 -0.02 -0.02 -0.03 -0.03 -0.02 -0.03 -0.03 -0.03 0 ...
##  $ gyros_arm_z         : num  -0.02 -0.02 -0.02 0 0 0 -0.02 0 0 -0.03 ...
##  $ accel_arm_x         : int  -288 -290 -289 -289 -289 -289 -288 -290 -288 -289 ...
##  $ accel_arm_y         : int  109 110 110 111 111 111 109 110 111 109 ...
##  $ accel_arm_z         : int  -123 -125 -126 -123 -122 -124 -122 -123 -123 -121 ...
##  $ magnet_arm_x        : int  -368 -369 -368 -374 -369 -372 -369 -366 -363 -367 ...
##  $ magnet_arm_y        : int  337 337 344 337 342 338 341 339 343 340 ...
##  $ magnet_arm_z        : int  516 513 513 506 513 510 518 509 520 509 ...
##  $ roll_dumbbell       : num  13.1 13.1 12.9 13.4 13.4 ...
##  $ pitch_dumbbell      : num  -70.5 -70.6 -70.3 -70.4 -70.8 ...
##  $ yaw_dumbbell        : num  -84.9 -84.7 -85.1 -84.9 -84.5 ...
##  $ total_accel_dumbbell: int  37 37 37 37 37 37 37 37 37 37 ...
##  $ gyros_dumbbell_x    : num  0 0 0 0 0 0 0 0 0 0 ...
##  $ gyros_dumbbell_y    : num  -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 ...
##  $ gyros_dumbbell_z    : num  0 0 0 0 0 0 0 0 0 0 ...
##  $ accel_dumbbell_x    : int  -234 -233 -232 -233 -234 -234 -232 -233 -233 -233 ...
##  $ accel_dumbbell_y    : int  47 47 46 48 48 46 47 47 47 48 ...
##  $ accel_dumbbell_z    : int  -271 -269 -270 -270 -269 -272 -269 -269 -270 -271 ...
##  $ magnet_dumbbell_x   : int  -559 -555 -561 -554 -558 -555 -549 -564 -554 -554 ...
##  $ magnet_dumbbell_y   : int  293 296 298 292 294 300 292 299 291 297 ...
##  $ magnet_dumbbell_z   : num  -65 -64 -63 -68 -66 -74 -65 -64 -65 -73 ...
##  $ roll_forearm        : num  28.4 28.3 28.3 28 27.9 27.8 27.7 27.6 27.5 27.1 ...
##  $ pitch_forearm       : num  -63.9 -63.9 -63.9 -63.9 -63.9 -63.8 -63.8 -63.8 -63.8 -64 ...
##  $ yaw_forearm         : num  -153 -153 -152 -152 -152 -152 -152 -152 -152 -151 ...
##  $ total_accel_forearm : int  36 36 36 36 36 36 36 36 36 36 ...
##  $ gyros_forearm_x     : num  0.03 0.02 0.03 0.02 0.02 0.02 0.03 0.02 0.02 0.02 ...
##  $ gyros_forearm_y     : num  0 0 -0.02 0 -0.02 -0.02 0 -0.02 0.02 0 ...
##  $ gyros_forearm_z     : num  -0.02 -0.02 0 -0.02 -0.03 0 -0.02 -0.02 -0.03 0 ...
##  $ accel_forearm_x     : int  192 192 196 189 193 193 193 193 191 194 ...
##  $ accel_forearm_y     : int  203 203 204 206 203 205 204 205 203 204 ...
##  $ accel_forearm_z     : int  -215 -216 -213 -214 -215 -213 -214 -214 -215 -215 ...
##  $ magnet_forearm_x    : int  -17 -18 -18 -17 -9 -9 -16 -17 -11 -13 ...
##  $ magnet_forearm_y    : num  654 661 658 655 660 660 653 657 657 656 ...
##  $ magnet_forearm_z    : num  476 473 469 473 478 474 476 465 478 471 ...
##  $ classe              : Factor w/ 5 levels "A","B","C","D",..: 1 1 1 1 1 1 1 1 1 1 ...
```

```r
# Remove the first 7 columns as they are not exercise movement related.
clean_train_set <- select(train_set, c(7:60))


# Clean the test_data the same way.
test_nas <- sapply(test_data, function(x) { sum(is.na(x))})
#table(test_nas)
clean_test_set <- test_data[, !names(test_data) %in% remove_cols]
str(clean_test_set)
```

```
## 'data.frame':	20 obs. of  60 variables:
##  $ X                   : int  1 2 3 4 5 6 7 8 9 10 ...
##  $ user_name           : Factor w/ 6 levels "adelmo","carlitos",..: 6 5 5 1 4 5 5 5 2 3 ...
##  $ raw_timestamp_part_1: int  1323095002 1322673067 1322673075 1322832789 1322489635 1322673149 1322673128 1322673076 1323084240 1322837822 ...
##  $ raw_timestamp_part_2: int  868349 778725 342967 560311 814776 510661 766645 54671 916313 384285 ...
##  $ cvtd_timestamp      : Factor w/ 11 levels "02/12/2011 13:33",..: 5 10 10 1 6 11 11 10 3 2 ...
##  $ new_window          : Factor w/ 1 level "no": 1 1 1 1 1 1 1 1 1 1 ...
##  $ num_window          : int  74 431 439 194 235 504 485 440 323 664 ...
##  $ roll_belt           : num  123 1.02 0.87 125 1.35 -5.92 1.2 0.43 0.93 114 ...
##  $ pitch_belt          : num  27 4.87 1.82 -41.6 3.33 1.59 4.44 4.15 6.72 22.4 ...
##  $ yaw_belt            : num  -4.75 -88.9 -88.5 162 -88.6 -87.7 -87.3 -88.5 -93.7 -13.1 ...
##  $ total_accel_belt    : int  20 4 5 17 3 4 4 4 4 18 ...
##  $ gyros_belt_x        : num  -0.5 -0.06 0.05 0.11 0.03 0.1 -0.06 -0.18 0.1 0.14 ...
##  $ gyros_belt_y        : num  -0.02 -0.02 0.02 0.11 0.02 0.05 0 -0.02 0 0.11 ...
##  $ gyros_belt_z        : num  -0.46 -0.07 0.03 -0.16 0 -0.13 0 -0.03 -0.02 -0.16 ...
##  $ accel_belt_x        : int  -38 -13 1 46 -8 -11 -14 -10 -15 -25 ...
##  $ accel_belt_y        : int  69 11 -1 45 4 -16 2 -2 1 63 ...
##  $ accel_belt_z        : int  -179 39 49 -156 27 38 35 42 32 -158 ...
##  $ magnet_belt_x       : int  -13 43 29 169 33 31 50 39 -6 10 ...
##  $ magnet_belt_y       : int  581 636 631 608 566 638 622 635 600 601 ...
##  $ magnet_belt_z       : int  -382 -309 -312 -304 -418 -291 -315 -305 -302 -330 ...
##  $ roll_arm            : num  40.7 0 0 -109 76.1 0 0 0 -137 -82.4 ...
##  $ pitch_arm           : num  -27.8 0 0 55 2.76 0 0 0 11.2 -63.8 ...
##  $ yaw_arm             : num  178 0 0 -142 102 0 0 0 -167 -75.3 ...
##  $ total_accel_arm     : int  10 38 44 25 29 14 15 22 34 32 ...
##  $ gyros_arm_x         : num  -1.65 -1.17 2.1 0.22 -1.96 0.02 2.36 -3.71 0.03 0.26 ...
##  $ gyros_arm_y         : num  0.48 0.85 -1.36 -0.51 0.79 0.05 -1.01 1.85 -0.02 -0.5 ...
##  $ gyros_arm_z         : num  -0.18 -0.43 1.13 0.92 -0.54 -0.07 0.89 -0.69 -0.02 0.79 ...
##  $ accel_arm_x         : int  16 -290 -341 -238 -197 -26 99 -98 -287 -301 ...
##  $ accel_arm_y         : int  38 215 245 -57 200 130 79 175 111 -42 ...
##  $ accel_arm_z         : int  93 -90 -87 6 -30 -19 -67 -78 -122 -80 ...
##  $ magnet_arm_x        : int  -326 -325 -264 -173 -170 396 702 535 -367 -420 ...
##  $ magnet_arm_y        : int  385 447 474 257 275 176 15 215 335 294 ...
##  $ magnet_arm_z        : int  481 434 413 633 617 516 217 385 520 493 ...
##  $ roll_dumbbell       : num  -17.7 54.5 57.1 43.1 -101.4 ...
##  $ pitch_dumbbell      : num  25 -53.7 -51.4 -30 -53.4 ...
##  $ yaw_dumbbell        : num  126.2 -75.5 -75.2 -103.3 -14.2 ...
##  $ total_accel_dumbbell: int  9 31 29 18 4 29 29 29 3 2 ...
##  $ gyros_dumbbell_x    : num  0.64 0.34 0.39 0.1 0.29 -0.59 0.34 0.37 0.03 0.42 ...
##  $ gyros_dumbbell_y    : num  0.06 0.05 0.14 -0.02 -0.47 0.8 0.16 0.14 -0.21 0.51 ...
##  $ gyros_dumbbell_z    : num  -0.61 -0.71 -0.34 0.05 -0.46 1.1 -0.23 -0.39 -0.21 -0.03 ...
##  $ accel_dumbbell_x    : int  21 -153 -141 -51 -18 -138 -145 -140 0 -7 ...
##  $ accel_dumbbell_y    : int  -15 155 155 72 -30 166 150 159 25 -20 ...
##  $ accel_dumbbell_z    : int  81 -205 -196 -148 -5 -186 -190 -191 9 7 ...
##  $ magnet_dumbbell_x   : int  523 -502 -506 -576 -424 -543 -484 -515 -519 -531 ...
##  $ magnet_dumbbell_y   : int  -528 388 349 238 252 262 354 350 348 321 ...
##  $ magnet_dumbbell_z   : int  -56 -36 41 53 312 96 97 53 -32 -164 ...
##  $ roll_forearm        : num  141 109 131 0 -176 150 155 -161 15.5 13.2 ...
##  $ pitch_forearm       : num  49.3 -17.6 -32.6 0 -2.16 1.46 34.5 43.6 -63.5 19.4 ...
##  $ yaw_forearm         : num  156 106 93 0 -47.9 89.7 152 -89.5 -139 -105 ...
##  $ total_accel_forearm : int  33 39 34 43 24 43 32 47 36 24 ...
##  $ gyros_forearm_x     : num  0.74 1.12 0.18 1.38 -0.75 -0.88 -0.53 0.63 0.03 0.02 ...
##  $ gyros_forearm_y     : num  -3.34 -2.78 -0.79 0.69 3.1 4.26 1.8 -0.74 0.02 0.13 ...
##  $ gyros_forearm_z     : num  -0.59 -0.18 0.28 1.8 0.8 1.35 0.75 0.49 -0.02 -0.07 ...
##  $ accel_forearm_x     : int  -110 212 154 -92 131 230 -192 -151 195 -212 ...
##  $ accel_forearm_y     : int  267 297 271 406 -93 322 170 -331 204 98 ...
##  $ accel_forearm_z     : int  -149 -118 -129 -39 172 -144 -175 -282 -217 -7 ...
##  $ magnet_forearm_x    : int  -714 -237 -51 -233 375 -300 -678 -109 0 -403 ...
##  $ magnet_forearm_y    : int  419 791 698 783 -787 800 284 -619 652 723 ...
##  $ magnet_forearm_z    : int  617 873 783 521 91 884 585 -32 469 512 ...
##  $ problem_id          : int  1 2 3 4 5 6 7 8 9 10 ...
```

```r
clean_test_set <- select(clean_test_set, c(7:60))
```


### Exploratory Analysis
Look at the frequency of each type of classe variable

```r
summary(clean_train_set$classe)
```

```
##    A    B    C    D    E 
## 3348 2279 2054 1930 2165
```

### Plot the frequencies

```r
plot(clean_train_set$classe, main = "Excercise Type Frequencies, 'classe' variable", xlab = "Exercise Type Groupings", ylab = "Count pe Exercise Type")
```

![plot of chunk unnamed-chunk-6](figure/unnamed-chunk-6-1.png) 

## Building the Predictive Model

### Sample Clean Data
Split the clean training data into 60% training and 40% test data

```r
clean_training <- createDataPartition(y = clean_train_set$classe, p=0.6, list=FALSE)
model_train_set <- clean_train_set[clean_training, ]
model_test_set <- clean_train_set[-training, ]
```

Using the Random Forest method and 4 fold cross validation, we will build a training model to predict the 'classe' on the remaining test set.

```r
model <-train(model_train_set$classe~., method = "rf", data = model_train_set, trControl = trainControl(method = "cv", number = 4, allowParallel = TRUE))

print(model)
```

```
## Random Forest 
## 
## 7067 samples
##   53 predictor
##    5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (4 fold) 
## 
## Summary of sample sizes: 5299, 5300, 5300, 5302 
## 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa      Accuracy SD  Kappa SD   
##    2    0.9830192  0.9785123  0.004262802  0.005397246
##   27    0.9903776  0.9878275  0.003396666  0.004297766
##   53    0.9845768  0.9804900  0.002743560  0.003472060
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 27.
```

### In Sample Accuracy
We run predict on the training set data with the train model we just built.
Then we calculate the In Sample Accuracy which is the accuracy of our prediciton model on the training data set.


```r
predictions <- predict(model, model_train_set)
confusionMatrix(predictions, model_train_set$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2009    0    0    0    0
##          B    0 1368    0    0    0
##          C    0    0 1233    0    0
##          D    0    0    0 1158    0
##          E    0    0    0    0 1299
## 
## Overall Statistics
##                                      
##                Accuracy : 1          
##                  95% CI : (0.9995, 1)
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
## Prevalence             0.2843   0.1936   0.1745   0.1639   0.1838
## Detection Rate         0.2843   0.1936   0.1745   0.1639   0.1838
## Detection Prevalence   0.2843   0.1936   0.1745   0.1639   0.1838
## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000
```
We get an accuracy of 100% on the training data.

### Out of Sample Accuracy
Run the model on the test set

```r
test_predictions <- predict(model, model_test_set)
confusionMatrix(test_predictions, model_test_set$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1334    1    0    0    0
##          B    0  916    3    0    0
##          C    0    1  819    1    0
##          D    0    0    0  766    2
##          E    0    0    0    1  879
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9981          
##                  95% CI : (0.9964, 0.9991)
##     No Information Rate : 0.2824          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9976          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9978   0.9964   0.9974   0.9977
## Specificity            0.9997   0.9992   0.9995   0.9995   0.9997
## Pos Pred Value         0.9993   0.9967   0.9976   0.9974   0.9989
## Neg Pred Value         1.0000   0.9995   0.9992   0.9995   0.9995
## Prevalence             0.2824   0.1944   0.1740   0.1626   0.1865
## Detection Rate         0.2824   0.1939   0.1734   0.1622   0.1861
## Detection Prevalence   0.2827   0.1946   0.1738   0.1626   0.1863
## Balanced Accuracy      0.9999   0.9985   0.9979   0.9984   0.9987
```
We get an accuracy of 99.47% on the testing data.

### Prediction Assignment
Run the prediction model on the clean data set that was kept separate during the training and testing phases.

```r
results <- predict(model, clean_test_set)
answers <- as.character(results)
print(answers)
```

```
##  [1] "B" "A" "B" "A" "A" "E" "D" "B" "A" "A" "B" "C" "B" "A" "E" "E" "A"
## [18] "B" "B" "B"
```


```r
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(answers)
```



