#!/usr/bin/ruby

require 'csv'

require 'nokogiri'
require 'open-uri'
#require 'oj'

base_sleep = 0
sleep_increment = 3
retries = 4

#http://www.nfl.com/liveupdate/game-center/2013090500/2013090500_gtd.json

base_url = "http://www.nfl.com"

nfl_team_schedules = CSV.open("csv/nfl_team_schedules.csv","r",{:col_sep => "\t", :headers => TRUE})

games = []
nfl_team_schedules.each do |game|
  year = game["season"].to_i
  game_url = game["game_url"]
  if (game_url==nil)
    next
  end
  game_id = game_url.split("/")[4] rescue nil
  if not(game_id==nil)
    games << [year,game_id]
  end
end

games.sort!
games.uniq!

team_game_base_url = "http://www.nfl.com/liveupdate/game-center/"

games_found = 0
games.each do |game|

  year = game[0]
  if (year<2009)
    next
  end
  game_id = game[1]

  sleep_time = base_sleep

  url = team_game_base_url+"#{game_id}/#{game_id}_gtd.json"

  p url

  #print "Sleep #{sleep_time} ... "
  sleep sleep_time

  tries = 0
  begin
    json_data = open(url).read
  rescue
    sleep_time += sleep_increment
    print "sleep #{sleep_time} ... "
    sleep sleep_time
    tries += 1
    if (tries > retries)
      next
    else
      retry
    end
  end

  #json = Oj.load(json_data)
  json_file = open("json/#{game_id}.json", "w") 
  #json_file.write(Oj.dump(json))
  json_file.write(json_data)
  json_file.close

  sleep_time = base_sleep
  games_found += 1

end
print "games found = #{games_found}\n"
