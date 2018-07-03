#------------------------------------------------------------------------------
#
#  This script will conduct the following analyses between 2 groups
#  - compute mean shape
#  - test hypothesis (Goodall's F-test)
#  - compute mean euclidean distance per landmark
#
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
# Load libraries
library(shapes)
library(foreach)
#library(reticulate)

# Load data
args <- commandArgs(trailingOnly = TRUE)
data_path <- args[1]
setwd(data_path)
l_grps <- list("g1_landmark_matrix", "g2_landmark_matrix")

#------------------------------------------------------------------------------
# Rearranges data from csv as R array of matrices 

# [toformat] modifies raw csv input into arrays of 2x68 matrices 
# of studied group
toformat <- function(x){
  lpt <- list()
  i <- ncol(x)
    cols <- (1:i)
    for (n in cols){
      pt = matrix(c(x[,n]), nrow=2, ncol=68, byrow=FALSE)
      pt <- t(pt)
      lpt[[n]] <- pt
    }
  fdat <- simplify2array(lpt)
  return(fdat)
}

# calls toformat function
prep <- function(csv_file){
  x <- read.csv(csv_file)
  x <- t(x)
  x <- x[2:137,]
  x <- toformat(x)
  return(x)
}

# calls prep function on the list of groups
loader <- function(l_grps){
  lr_grps <- list()
  for (i in l_grps){
    index <- match(i, l_grps)
    grp <- paste(i,'csv', sep = ".")
    grp <- prep(grp)
    lr_grps[[index]] <- grp
  }
  return(lr_grps)
}

# list containing groups (arrays of matrices)
lr_grps <- loader(l_grps)


#------------------------------------------------------------------------------
# Run statistical analysis

# simplified GPA
gpa <- function(grp){
  grp_p <- procGPA(grp, eigen2d=TRUE)
  return(grp_p)
}

# runs the generalized procustre analysis on groups
lp_grps <- list()
proc <- function(lr_grps, lp_grps, x){
  i <- x
  y <- length(lr_grps)
  if (i < y){
    grp_p <- gpa(lr_grps[[i]])
    lp_grps[[length(lp_grps)+1]] <- grp_p
    i <- i + 1
    proc(lr_grps, lp_grps, i)
  }
  else{
    grp_p <- gpa(lr_grps[[y]])
    lp_grps[[length(lp_grps)+1]] <- grp_p
    return(lp_grps)
  }
}

# computes landmark distance from a predetermined baseline
distpt <- function(x2, base){
  ldist <- list()
  eucdist <- function(x1, x2) sqrt(sum((x1-x2)^2))
  d <- dim(x2)
  i <- d[3]
  for (n in 1:i){
    pt <- x2[,,n]
    dist <- foreach(z = 1:nrow(pt), .combine = c) %do% 
	    eucdist(base[z,], pt[z,])
    ldist[[n]] <- dist
    }
  ldist <- t(data.frame(simplify2array(ldist)))
  return(ldist)
}

# computes mean distance per landmark for each group
mdist_landmark <- function(all_dist){
  mdist_ldmk <- list()
  dimensions <- dim(all_dist)
  pts <- dimensions[2]
  ldmk <- dimensions[1]
  for (n in 1:pts){
    mn <- mean(all_dist[,n])
    mdist_ldmk[n] <- mn
  }
  mdist_ldmk <- simplify2array(mdist_ldmk)
  mdist_ldmk <- t(data.frame(mdist_ldmk))
  return(mdist_ldmk)
}

# runs mean landmark distance analysis 
analysis <- function(lr_grps){
  lmd_grps <- list()
  base <- gpa(lr_grps[[1]])
  base <- base$mshape
  for (i in lr_grps){
    index <- match(i, lr_grps)
    gpa_grp <- gpa(i)
    distpt_grp <- distpt(gpa_grp$rotated, base)
    mdist_grp <- mdist_landmark(distpt_grp)
    lmd_grps[[length(lmd_grps)+1]] <- mdist_grp
  }
  return(lmd_grps)
}

# list of mean landmark distance per group
lmd_grps <- analysis(lr_grps)
# list of results from procustrean analysis
lp_grps <- proc(lr_grps, lp_grps, 1)
# test mean shape (goodall statistical test)
tms <- testmeanshapes(lr_grps[[1]], lr_grps[[2]], resamples=10, replace=TRUE)


#------------------------------------------------------------------------------
# plot mean euclidean distance    
a <- simplify2array(lmd_grps[1])
b <- simplify2array(lmd_grps[2])
grp1 <- apply(a, 2, mean)
grp2 <- apply(b, 2, mean)
all_mdist  <- cbind(grp1, grp2)
all_mdist  <- t(all_mdist)
colours <- c('red', 'green')
# save plot as png image
png(filename="histo.png", res=720, width=16, height=9, units="in", pointsize=6)
barplot(
  as.matrix(all_mdist), 
  main='Mean Euclidean Distance of Each Landmark', 
  names.arg=c(1:68),
  ylab='Distance',
  xlab='Landmark',
  beside=TRUE, 
  col=colours,
  legend=rownames(all_mdist)
)
dev.off()



# print out ouput of the tests
st0 <- noquote("--------------------------------------------------------------------------------")
st1 <- noquote("Summary of Mean Euclidean Distance:")
st2 <- noquote("Group 1:")
st3 <- noquote(paste(noquote("Mean: "), mean(grp1), noquote("| Standard Deviation: "), sd(grp1)))
st4 <- noquote("Group 2:")
st5 <- noquote(paste(noquote("Mean: "), mean(grp2), noquote("| Standard Deviation: "), sd(grp2)))
st6 <- noquote(paste(noquote("Goodall Statistical Test P-Value: "), tms$G.pvalue))
print(st0)
print(st6)
print(st0)
print(st1)
print(st2)
print(st3)
print(st4)
print(st5)
print(st0)
