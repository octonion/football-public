#!/usr/bin/ruby1.9.3

require "csv"
require "matrix"

class Matrix
  def []=(i, j, x)
    @rows[i][j] = x
  end
end

c = CSV.open("nfl_picks.csv","r")

m=1
c.each do |row|
  r1 = row[1].split("/")
  r2 = row[2].split("/")
  r1.each do |r|
    if (r.to_i > m)
      m=r.to_i
    end
  end
  r2.each do |r|
    if (r.to_i > m)
      m=r.to_i
    end
  end
end

a = Matrix.zero(m)
c = CSV.open("nfl_picks.csv","r")
c.each_with_index do |row,i|
  r1 = row[1].split("/")
  r2 = row[2].split("/")
  r1.each do |r|
    a[i,r.to_i-1] += 1
  end
  r2.each do |r|
    a[i,r.to_i-1] -= 1
  end  
end

p a.to_a.flatten

