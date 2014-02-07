library("MASS")

m <- matrix(scan(file="m.csv"),329,329)

q <- qr(m)
q$rank

d <- diag(ginv(m))
e <- cbind(round((3000*d)/d[1]))
e
