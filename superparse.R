superparse <- function(Lines){
  Lines <- strsplit(Lines, ',')
  
  cuts <- which(sapply(Lines, length)==4)
  cuts <- cuts[2:length(cuts)]
  cuts <- c(cuts, length(Lines))
  
  ranges <- lapply(1:(length(cuts)-1), 
                   function(x){return((cuts[x]+1):(cuts[x+1]))})
  ranges <- ranges[lapply(ranges, length) > 2]
  Lines <- lapply(ranges, function(x){Lines[x]})
  Matrices <- lapply(Lines, make.matrix)
  GO <- lapply(Matrices, cat.accuracy)
  return(GO)
}

make.matrix <- function(Lines){
  target.l <- length(Lines[[1]])
  Lines <- Lines[lapply(Lines, length)==target.l]
  return(matrix(as.integer(unlist(Lines)), ncol=length(Lines[[1]]), byrow=T))
}

cat.accuracy <- function(mx){
  acc.vector = c()
  for(i in 1:dim(mx)[1]){
    acc.vector <- c(acc.vector, mx[i,i]/sum(mx[i,]))
  }
  return(acc.vector)
}

simpleparse <- function(lns){
  lns <- strsplit(lns, ',')
  lns <- q[which(sapply(q, length)==4)]
  nms <- lns[[1]]
  lns <- lns[grep('[0-9]', lns)]
  lns <- lapply(lns, as.numeric)
  df <- as.data.frame(do.call(rbind, lns), stringsAsFactors=F)
  names(df) <- nms
  return(df)
}