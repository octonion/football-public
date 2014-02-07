#!/usr/bin/ruby1.9.3

require "csv"

c = CSV.open("nfl_picks.csv","r")
d = CSV.open("nfl_picks_fixed.csv","w")

m1=0
m2=0
c.each do |row|
  p row
  m = row[1].split("/").size
  if (m>m1)
    m1=m
  end
  m = row[2].split("/").size
  if (m>m2)
    m2=m
  end
end

if (m1>m2)
  m=m1
else
  m=m2
end

c = CSV.open("nfl_picks.csv","r")

c.each do |row|
  p1 = row[1]
  s = p1.split("/").size
  if (s<m)
    p1 = p1+"/B"*(m-s)
  end
  p2 = row[2]
  s = p2.split("/").size
  if (s<m)
    p2 = p2+"/B"*(m-s)
  end
  r = [row[0],p1,p2]
  d << r
end
