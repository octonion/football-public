#!/usr/bin/ruby

require 'csv'

require 'nokogiri'
require 'open-uri'

base_sleep = 0
sleep_increment = 3
retries = 4

#http://www.nfl.com/teams/newengland%20patriots/schedule?team=NE&season=2012&seasonType=REG

#http://www.nfl.com/liveupdate/game-center/2013090500/2013090500_gtd.json
#http://www.nfl.com/scores/2013/REG1

base_url = "http://www.nfl.com"

nfl_team_standings = CSV.open("csv/nfl_team_standings.csv","r",{:col_sep => "\t", :headers => TRUE})

nfl_team_schedules = CSV.open("csv/nfl_team_standings.csv","w",{:col_sep => "\t"})

# Header for team standings file

nfl_team_standings << ["season", "team_name", "status", "team_url", "wins", "losses", "ties", "win_percentage", "points_for", "points_against", "net_points", "touchdowns", "home_record", "road_record", "division_record", "division_win_percentage", "conference_record", "conference_win_percentage", "nonconference_record","winning_streak","last_5_record"]

#AFC East Team	W	L	T	 Pct	PF	PA	Net Pts	TD	Home	Road	Div	Pct	Conf	Pct	Non-Conf	Streak	Last 5

#base_url = "http://www.nfl.com/standings?category=div&season=2012-REG&split=Overall"

team_standings_url = "http://www.nfl.com/standings?category=div"

team_standings_xpath = '//*[@class="data-table1"]/tbody/tr'

#(2012..2013).each do |year|
(1920..2013).each do |year|

  sleep_time = base_sleep
  url = team_standings_url+"&season=#{year}-REG&split=Overall"
  
  #print "Sleep #{sleep_time} ... "
  sleep sleep_time

  tries = 0
  begin
    doc = Nokogiri::HTML(open(url))
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

  sleep_time = base_sleep

  #  print "#{year} #{team_name} ..."

  doc.xpath(team_standings_xpath).each do |team|

    row = [year]
    team.xpath("td").each_with_index do |field,j|
      text = field.text.strip rescue nil
      case j
      when 0
        parts = text.split("- ")
        if (parts.size>1)
          status = parts[0].strip rescue nil
          team_name = parts[1].strip rescue nil
          link = field.xpath("a").first
          team_url = (base_url+link.attributes["href"].text) rescue nil
        else
          status = nil
          team_name = parts[0].strip rescue nil
          link = field.xpath("a").first
          team_url = (base_url+link.attributes["href"].text) rescue nil
        end
        row += [team_name,status,team_url]
      else
        row << field.text.strip rescue nil
      end
    end
    if (row.size>4) and not(row[4]=="W")
      nfl_team_standings << row
    end

  end

end
