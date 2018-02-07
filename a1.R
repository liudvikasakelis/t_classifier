args <- commandArgs(TRUE)
# args <- c('data.csv', '50', '10')

### USING category_mappings.txt FROM dirname() TO RE-MAP CATEGORIES

infile <- args[1]
# mode <- args[2]
cutoff.1 <- as.integer(args[2]) 
cutoff.2 <- as.integer(args[3])
if (is.na(cutoff.1)){
  cutoff.1 <- 100
  cat('defaulting to', cutoff.1, 'for cutoff.1 (min transations in category)\n')
}

if (is.na(cutoff.2)){
  cutoff.2 <- 10
  cat('defaulting to', cutoff.2, 'for cutoff.2 (min people in category)\n')
}

q <- read.csv(infile, as.is=T, na.string="NULL", header=T)
q$operation_date_less_than_2016_01_01 <- NULL
### SAFE???
names(q)[grep('inout', names(q))] <- 'operationinout' 
###

others <- c(2000, 3000)

q$y1[q$y1 %in% others] <- NA
q$y2[q$y2 %in% others] <- NA

Encoding(q$paymentpurpose) <- "UTF-8" # IS NEEDED
q$paymentpurpose <- gsub('\\s+', ' ', q$paymentpurpose)

q$y <- q$y1
na.mask <- is.na(q$y)
q$y[na.mask] <- q$y2[na.mask]

# Remove NAs
q <- q[!is.na(q$y),]
q <- q[!is.na(q$operationdate),]
q$payerbank[is.na(q$payerbank)] <- 9
q$receiverbank[is.na(q$receiverbank)] <- 9

# Remove old Y
q$y1 <- NULL
q$y2 <- NULL
cat(nrow(q), 'rows\n')

all.cats <- unique(q$y)
all.cats <- all.cats[!is.na(all.cats)]
cat(length(all.cats), 'categories\n')

small.cats <- c() # categories with less transactions than cutoff.1
for(category in all.cats){
  if(sum(q$y == category) %in% 1:cutoff.1){
    small.cats <- c(small.cats, category)
  }
}

cat('Small categories (', length(small.cats), ') :\n', small.cats, "\n")
cat(sum(q$y %in% small.cats) , "small category transactions to be removed\n")
q <- q[!(q$y %in% small.cats),]
cat(nrow(q), "rows left\n")

ppl.cats <- c() # categories with less people than cutoff.2
for(category in all.cats){
  if(length(unique(q$StatementHolderId[q$y == category])) < cutoff.2){
    ppl.cats <- c(ppl.cats, category)
  }
}
cat("Few people categories (", length(ppl.cats), ")\n", ppl.cats, "\n")
cat(sum(q$y %in% ppl.cats) , "few people transactions to be removed\n")
q <- q[!(q$y %in% ppl.cats),]
cat(nrow(q), "rows left\n")

### MAPPING
cat.map <- read.csv(paste(dirname(infile), 'category_mappings.txt', sep='/'),
                    header=T, as.is=T)
for(category in cat.map$original){
  mapped.cat <- cat.map$mapped[cat.map$original==category]
  q$y[q$y == category] <- mapped.cat
}
###

### TIME 
q$operationdate <- as.integer(as.Date(q$operationdate))
###

### Paste payerbank, receiverbank, operationinout
q$paymentpurpose <- paste(as.character(q$operationinout), 
                          as.character(q$payerbank), 
                          as.character(q$receiverbank), 
                          q$paymentpurpose, sep='') 
###

all.ppl <- unique(q$StatementHolderId)
all.cats <- unique(q$y)
for(i in 1:100){
  failed.cats <- c()
  test.ppl <- sample(all.ppl, length(all.ppl)/2)
  test <- q[q$StatementHolderId %in% test.ppl,]
  train <- q[!(q$StatementHolderId %in% test.ppl),]
  cat(nrow(train), ":", nrow(test), "\n")
  for(category in all.cats){
    test.sum <- sum(test$y == category)
    train.sum <- sum(train$y == category)
    ratio = min(test.sum, train.sum) / max(test.sum, train.sum)
    if (ratio == 0 || train.sum/test.sum < 0.5){
      failed.cats <- c(failed.cats, category)
      cat(category, '->', train.sum, ':', test.sum, '\n')
    }
  }
  if (length(failed.cats) == 0){
    cat('dingdingdingding\n')
    break
  } 
  cat(failed.cats, '\n')
}

train.ppl <- unique(train$StatementHolderId)
train.cats <- unique(train$y)
cat(length(train.cats), "train cats\n")
for(i in 1:100){
  train.ppl <- train.ppl[sample(length(train.ppl))]
  folds <- cut(seq(1, length(train.ppl)), breaks=10, labels=FALSE)
  failed.cats <- c()
  for(j in 1:10){
    XVtest.indices <- which(folds == j, arr.ind=T)
    XVtest.ppl <- train.ppl[XVtest.indices]
    XVtest <- train[train$StatementHolderId %in% XVtest.ppl,]
    XVtrain <- train[!(train$StatementHolderId %in% XVtest.ppl),]
    # cat(i, j, nrow(XVtest), nrow(XVtrain), "\n")
    for(category in train.cats){
      if(!category %in% XVtrain$y){
        failed.cats <- c(failed.cats, category)
      }
    }
  }
  if(length(failed.cats) == 0){
    cat('dingdingdingding\n')
    break
  }
  cat(failed.cats, '\n')
}

train.ppl <- data.frame(folds, train.ppl)
names(train.ppl) <- c("fold", "StatementHolderId")
train <- merge(train, train.ppl, by="StatementHolderId", all.x=T)


### Writeout NOT respecting cutoff.3 date

train.out <- file.path(dirname(infile), paste('train', basename(infile), sep='.'))
test.out <- file.path(dirname(infile), paste('test', basename(infile), sep='.'))
write.table(train, train.out, row.names=F, col.names=T, sep=",", 
            qmethod="double", fileEncoding="UTF-8")
write.table(test, test.out, row.names=F, col.names=T, sep=",", 
            qmethod="double", fileEncoding="UTF-8") 





