library(data.table)
library(lightgbm)
library(caret)

LGB_CV_Predict <- function(lgb_cv, data, num_iteration = NULL, folds=NULL, type=c("cv", "test")) {
  require(foreach)
  if (is.null(num_iteration)) {
    num_iteration <- lgb_cv$best_iter
  }
  if (type=="cv"){
    print("create cross validation predictions")
    pred_mat <- foreach::foreach(i = seq_along(lgb_cv$boosters), .combine = "c", .packages=c("data.table","lightgbm")) %do% {
      lgb_tree <- lgb_cv$boosters[[i]][[1]]
      predict(lgb_tree, 
              data[folds[[i]],], 
              num_iteration = num_iteration, 
              rawscore = FALSE, predleaf = FALSE, header = FALSE, reshape = FALSE)
    }
    
    as.double(pred_mat)[order(unlist(folds))]
    
  } else if (type=="test"){
    print("create test set predictions")
    pred_mat <- foreach::foreach(i = seq_along(lgb_cv$boosters), .combine = "+", .packages=c("data.table","lightgbm")) %do% {
      lgb_tree <- lgb_cv$boosters[[i]][[1]]
      predict(lgb_tree, 
              data, 
              num_iteration = lgb_cv$best_iter, 
              rawscore = FALSE, predleaf = FALSE, header = FALSE, reshape = FALSE)
    }
    as.double(pred_mat)/length(lgb_cv$boosters)
  }
}


t1 <- fread("F:/Dropbox/Kaggle Competition/Santander Customer Transaction Prediction/train_new.csv")
s1 <- fread("F:/Dropbox/Kaggle Competition/Santander Customer Transaction Prediction/test_new.csv")
t1[,":="(idx=.I,
         filter=0)]
s1[,":="(idx=.I,
         target=-1,
         filter=2)]
var_cols <- grep("var_", colnames(t1), value=T)

target0 <- t1[target==0]
target1 <- t1[target==1]

#set.seed(99999)
#for (col in var_cols){
#  set(target0, j=col, value=sample(target0[[col]], replace=F))
#  set(target1, j=col, value=sample(target1[[col]], replace=F))
#}
  
t1_new <- rbind(target0, target1)
setorder(t1_new, idx)
  
ts1 <- rbind(t1_new, s1)
set.seed(99999)

# k from 10 to 15

cvFoldsList <- createFolds(as.factor(ts1[filter==0, target]), k=10)

varnames <- setdiff(colnames(ts1), c("ID_code","target", "filter","fold","idx"))
dtrain <- lgb.Dataset(data.matrix(ts1[filter==0,varnames,with=F]), label=ts1[filter==0, target], free_raw_data = FALSE)


# num_leaves from 3 to 4
# learning rate from 0.01 to 0.05

params <- list(objective = "binary", 
               boost="gbdt",
               metric="auc",
               boost_from_average="false",
               num_threads=4,
               learning_rate = 0.05,
               num_leaves = 4,
               max_depth=-1,
               tree_learner = "serial",
               feature_fraction = 0.04,
               bagging_freq = 5,
               bagging_fraction = 0.4,
               min_data_in_leaf = 80,
               min_sum_hessian_in_leaf = 10.0,
               verbosity = 1)

tme <- Sys.time()

# nrounds from 2000000

lgb1 <- lgb.cv(params,
               dtrain,
               nrounds=1000000,
               folds=cvFoldsList,
               early_stopping_rounds = 3000,
               eval_freq=1000,
               seed=99999)
Sys.time() - tme
  
test_preds <- LGB_CV_Predict(lgb1, data.matrix(ts1[filter==2, varnames, with=F]), type="test")
dt <- data.table(ID_code=ts1[filter==2, ID_code], target=test_preds)

  
fwrite(dt, paste0("F:/Dropbox/Kaggle Competition/Santander Customer Transaction Prediction/lgb_submission_new_no_shuffle_more.csv"))
  














