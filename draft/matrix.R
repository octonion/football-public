
library("MASS")

m <- matrix(scan(file="m.csv"),329,329)
#n <- m[1:100,1:100]
#n <- n[-2,]
#n <- n[,-2]
#n <- n[-34,]
#n <- n[,-34]
q <- qr(m)
q$rank
d <- diag(ginv(m))
e <- cbind(round((3000*d)/d[1]))
e
