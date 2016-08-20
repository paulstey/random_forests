library(randomForest)

dset <- read.csv("./data/concrete_data.csv")


system.time(expr = {
    fm1 <- randomForest(comp_str ~ ., data = dset, mtry = 3, importance = TRUE, ntree = 50)
    }
)

