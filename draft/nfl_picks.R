
picks <- read.csv(file="nfl_picks.csv",header=TRUE)

picks1 <- strsplit(as.vector(picks$picks1),"/")
picks2 <- strsplit(as.vector(picks$picks2),"/")
l <- length(picks1)

gun <- function(i,beta) {
    if (i==1) { g <- 3000 };
    if (i==2) { g <- 2600 };
    if (i==3) { g <- 2200 };
    if (i==4) { g <- 1800 };
    if (i>4) { g <- 1800*exp(-beta*(i-4)) };
return(g)
}

beta <- 0.033
gun(as.integer(unlist(picks1[1])),beta)

fun <- function(beta) {
    s <- 0
    for(i in 1:length(picks1)) {
    s <- s+(sum(gun(as.integer(unlist(picks1[i])),beta))-sum(gun(as.integer(unlist(picks2[i])),beta)))^2}
    return(s)
}

fun(0.03)

o <- optimize(fun,c(0.01,0.10),tol=0.00001)
m <- o$minimum

v <- c(4:200)
1800*exp(-(v-4)*m)
