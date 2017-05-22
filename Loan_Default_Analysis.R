loan <- read.csv(file.choose())
loanV1 <- loan
str(loanV1)

#categorizing the target variable
loanV1$loan_isdefault <- ifelse(loanV1$loan_status=="Default",1,
                                ifelse(loanV1$loan_status=="Charged Off",1,
                                        ifelse(loanV1$loan_status=="Late (16-30 days)",1,
                                               ifelse(loanV1$loan_status=="Late (31-120 days)",1,
                                                      ifelse(loanV1$loan_status=="Does not meet the credit policy. Status:Charged Off",1,0)
                                                      )
                                        )
                                )
)
                                

summary(loanV1)

#removing certain columns due to poor data quality 
loanV1$loan_status <- NULL  
loanV1$url <- NULL
loanV1$desc <- NULL
loanV1$zip_code <- NULL
loanV1$mths_since_last_delinq <- NULL
loanV1$mths_since_last_record <- NULL
loanV1$mths_since_last_major_derog <- NULL
loanV1$annual_inc_joint <- NULL
loanV1$dti_joint <- NULL
loanV1$tot_coll_amt <- NULL
loanV1$tot_cur_bal <- NULL
loanV1$open_acc_6m <- NULL
loanV1$open_il_6m <- NULL
loanV1$open_il_12m <- NULL
loanV1$open_il_24m <- NULL
loanV1$mths_since_rcnt_il <- NULL
loanV1$total_bal_il <- NULL
loanV1$il_util <- NULL
loanV1$open_rv_12m <- NULL
loanV1$open_rv_24m <- NULL
loanV1$max_bal_bc <- NULL
loanV1$all_util <- NULL
loanV1$total_rev_hi_lim <- NULL
loanV1$inq_fi <- NULL
loanV1$total_cu_tl <- NULL
loanV1$inq_last_12m  <- NULL
loanV1$title <- NULL
loanV1$emp_title <- NULL
loanV1$next_pymnt_d <- NULL
loanV1$issue_d <- NULL
loanV1$title <- NULL
loanV1$earliest_cr_line <- NULL
loanV1$inq_last_6mths <- NULL
loanV1$recoveries <- NULL
loanV1$collection_recovery_fee <- NULL
loanV1$last_pymnt_d <- NULL
loanV1$last_pymnt_amnt <- NULL
loanV1$last_credit_pull_d <- NULL
loanV1$collections_12_mths_ex_med <- NULL
loanV1$verification_status_joint <- NULL
loanV1$application_type <- NULL
loanV1$policy_code <- NULL
loanV1$id <- NULL
loanV1$member_id <- NULL

#removing n/a values
sum(ifelse(loanV1$emp_length=="n/a",1,0))
loanV1$emp_length = ifelse(loanV1$emp_length == "n/a", 0, loanV1$emp_length)

#replacing NA values with mean
which.max(loanV1$dti)
loanV1$dti[loanV1$dti==9999] <- 18.16

loanV1$acc_now_delinq[is.na(loanV1$acc_now_delinq)] <- round(mean(loanV1$acc_now_delinq, na.rm = TRUE))
loanV1$total_acc[is.na(loanV1$total_acc)] <- round(mean(loanV1$total_acc, na.rm = TRUE))
loanV1$revol_util[is.na(loanV1$revol_util)] <- round(mean(loanV1$revol_util, na.rm = TRUE))
loanV1$pub_rec[is.na(loanV1$pub_rec)] <- round(mean(loanV1$pub_rec, na.rm = TRUE))
loanV1$open_acc[is.na(loanV1$open_acc)] <- round(mean(loanV1$open_acc, na.rm = TRUE))
loanV1$delinq_2yrs[is.na(loanV1$delinq_2yrs)] <- round(mean(loanV1$delinq_2yrs, na.rm = TRUE))
loanV1$annual_inc[is.na(loanV1$annual_inc)] <- round(mean(loanV1$annual_inc, na.rm = TRUE))


#Performing sampling to get 1 lakh rows 
set.seed(1234)
loanV2 <-loanV1[c(1:100000),]
loanV2<-loanV2[sample(nrow(loanV2)),]
loanV2$loan_isdefault <- as.factor(loanV2$loan_isdefault)


require(caTools)
set.seed(1234)
split<-sample.split(loanV2$loan_isdefault,SplitRatio = 0.7)
train1<-subset(loanV2,split==TRUE)
test1<-subset(loanV2,split==FALSE)

library(ggplot2)
ggplot(loanV2,aes(loanV2$loan_isdefault))+geom_bar(color="blue")

#performing oversampling to reduce the skewness of the target variable

require(ROSE)
loanV3 <- ovun.sample(loan_isdefault ~ ., data = loanV2, method = "over",p=0.3)$data
table(loanV3$loan_isdefault)

ggplot(loanV3,aes(loanV3$loan_isdefault))+geom_bar(color="blue")

require(caTools)
set.seed(1234)
split<-sample.split(loanV3$loan_isdefault,SplitRatio = 0.7)
train2<-subset(loanV3,split==TRUE)
test2<-subset(loanV3,split==FALSE)

require ("klaR")
require ("caret")
require ("e1071")
require (rpart)

#naive
model1<- naiveBayes(loan_isdefault ~ . , data=train2)
prediction1 <- predict(model1, test2)
confusionMatrix(test2$loan_isdefault, prediction1 )

#rpart
model2<-rpart(loan_isdefault ~ . , data=train2, cp=0.01)
prediction2 <- predict(model2, newdata=test2, type='class')
confusionMatrix(test2$loan_isdefault, prediction2 )
plot(model2)
text(model2, use.n = T, digits = 5, cex = 0.5)

caret::varImp(model2)

#gbm
require (gbm)
model3<- gbm(loan_isdefault ~ . , 
             distribution = "multinomial", data=train2, n.trees=100, cv.folds = 5, interaction.depth=6, shrinkage=0.01)
prediction3 = predict(model3, newdata=test2, n.trees=100, type="response")
pred_class <- apply(prediction3,1,which.max)
table(pred_class, test2$loan_isdefault)



#cross validation
require(caret)
ctrl <- trainControl(method="cv", number=10)
rpCVFit <- train(loan_isdefault ~ ., data=train2, method="rpart", trControl = ctrl, cp=0.01)
rpCVFit


#Create 5 equally size folds
folds <- cut(seq(1,nrow(loanV2)),breaks=5,labels=FALSE)
outputData = 0
#Perform 5 fold cross validation for naive bayes
for(i in 1:5){
  #Segment your data by fold using the which() function 
  testIndexes <- which(folds==i,arr.ind=TRUE)
  testData <- train2[testIndexes, ]
  trainData <- train2[-testIndexes, ]
  model1<- naiveBayes(loan_isdefault ~ . , data=trainData)
  prediction1 <- predict(model1, testData)
  misClassifyError = mean(prediction1 != testData$loan_isdefault)
  Accuracy = 1-misClassifyError 
  outputData[i] = Accuracy
}
head(outputData, 5)
summary(outputData)

install.packages("DescTools")

Desc(loanV2$loan_amnt, main = "Loan amount distribution", plotit = TRUE)
library(DescTools)

hist(loanV2$loan_amnt, xlab = "Loan Amount", main = "Distribution of Loan Amount")







