library(caret)
library(dplyr)
library(randomForest)

#getwd()

training_data_url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
test_data_url <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

training_data <- read.csv("PML/pml-training.csv", na.strings = c("NA", "#DIV/0!", ""))
test_data <- read.csv("PML/pml-testing.csv", na.strings = c("NA", "#DIV/0!", ""))

# split the training data for 60% training and 40% testing
training <- createDataPartition(y = training_data$classe, p=0.6, list=FALSE)
train_set <- training_data[training, ]
test_set <- training_data[-training, ]

# Remove Columns with lots of NA Values
sum(is.na(train_set))
nas <- sapply(train_set, function(x) { sum(is.na(x))})
table(nas)
# We can see that 60 columns dont have any NAs, while the others contain many (over 11,000 for the train_set). Lets remove those.
remove_cols <- names(nas[nas > 0])
train_set <- train_set[, !names(train_set) %in% remove_cols]
str(train_set)

# Remove the first 7 columns as they are not exercise movement related.
clean_train_set <- select(train_set, c(7:60))

#
# Clean the test_set the same way.
test_nas <- sapply(test_data, function(x) { sum(is.na(x))})
#table(test_nas)
clean_test_set <- test_data[, !names(test_data) %in% remove_cols]
str(clean_test_set)
clean_test_set <- select(clean_test_set, c(7:60))


#Exploratory Analysis
#Look at the frequency of each type of classe variable

summary(clean_train_set$classe)

#Plot the frequencies
plot(clean_train_set$classe, main = "Excercise Type Frequencies, 'classe' variable", xlab = "Exercise Type Groupings", ylab = "Count pe Exercise Type")




####
clean_training <- createDataPartition(y = clean_train_set$classe, p=0.6, list=FALSE)
model_train_set <- clean_train_set[clean_training, ]
model_test_set <- clean_train_set[-training, ]
# Machine Learning Model
# Using the Random Forest method we will build a machine lerning model to predict the 'classe' on the remaining test set.

model <-train(model_train_set$classe~., method = "rf", data = model_train_set, trControl = trainControl(method = "cv", number = 4, allowParallel = TRUE))

predictions <- predict(model, model_train_set)
confusionMatrix(predictions, model_train_set$classe)

test_predictions <- predict(model, model_test_set)
confusionMatrix(test_predictions, model_test_set$classe)


results <- predict(model, clean_test_set)
answers <- as.character(results)
print(answers)

pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(answers)