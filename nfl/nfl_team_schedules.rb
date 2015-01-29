#!/usr/bin/ruby

require 'csv'

require 'nokogiri'
require 'open-uri'

base_sleep = 0
sleep_increment = 3
retries = 4

#http://www.nfl.com/liveupdate/game-center/2013090500/2013090500_gtd.json
#http://www.nfl.com/scores/2013/REG1

base_url = "http://www.nfl.com"

nfl_team_standings = CSV.open("csv/nfl_team_standings.csv","r",{:col_sep => "\t", :headers => TRUE})

nfl_team_schedules = CSV.open("csv/nfl_team_schedules.csv","w",{:col_sep => "\t"})

# Header for team standings file

nfl_team_schedules << ["season", "team_id", "week", "date", "away_team_id", "away_team_url", "away_team_score", "home_team_id", "home_team_url", "home_team_score", "game_status", "game_url", "attendance", "top_passer", "top_passer_url", "top_passer_text", "top_rusher", "top_rusher_url", "top_rusher_text", "top_receiver", "top_receiver_url", "top_receiver_text"]

#Wk	Date	Game	Time (ET)	Attendance	Top Passer	Top Rusher	Top Receiver

team_schedules_url = "http://www.nfl.com/teams/"

#team_schedules_xpath = '//*[@id="team-stats-wrapper"]/table[3]/tr[position()>2]'
team_schedules_xpath = '//table[@class="data-table1"]/tr[position()>2]'

nfl_team_standings.each do |team_season|

  team_id = team_season["team_id"]
  year = team_season["season"].to_i
  if (year<2001)
    next
  end
  sleep_time = base_sleep

  url = team_schedules_url+"schedule?team=#{team_id}&season=#{year}&seasonType=REG"

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

  doc.xpath(team_schedules_xpath).each do |game|

    row = [year,team_id]
    game.xpath("td").each_with_index do |field,j|
      # See String#encode
      text = field.text.strip rescue nil
      case j
      when 2
        values = field.xpath("a").flat_map do |node|
           [(node.text.strip rescue nil),
            (base_url+node.attributes["href"].text rescue nil),
            (node.next_sibling.text.gsub("@","").strip rescue nil)]
#          [(n.previous_sibling.text.strip rescue nil),
        end
        row += values
      when 3
        values = field.xpath("a").flat_map do |node|
           [(node.text.strip rescue nil),
            (base_url+node.attributes["href"].text rescue nil)]
#          [(n.previous_sibling.text.strip rescue nil),

        end
        row += values
      when 5..7
        values = field.xpath("a").flat_map do |node|
           [(node.text.strip rescue nil),
            (base_url+node.attributes["href"].text rescue nil),nil]
#            (node.next_sibling.next_sibling.text.strip rescue nil)]
#          [(n.previous_sibling.text.strip rescue nil),
        end
        row += values
      else
        row += [text]
      end

    end

    nfl_team_schedules << row


#      text = field.text.gsub("\t"," ").gsub("\r\n"," ").gsub("\r","").gsub("\n","").strip rescue nil
      
#      encoding_options = {
#        :invalid           => :replace,  # Replace invalid byte sequences
#        :undef             => :replace,  # Replace anything not defined in ASCII
#        :replace           => '',        # Use a blank for those replacements
#        :universal_newline => true       # Always break lines with \n
#      }
      #text = text.encode(Encoding.find('ASCII'), encoding_options)

#      case j
#      when 2
        #parts = text.split("\t")
=begin
        if (parts.size>1)
          status = parts[0].strip rescue nil
          game_name = parts[1].strip rescue nil
          link = field.xpath("a").first
          game_url = (base_url+link.attributes["href"].text) rescue nil
        else
          status = nil
          game_name = parts[0].strip rescue nil
          link = field.xpath("a").first
          game_url = (base_url+link.attributes["href"].text) rescue nil
        end
=end
#        row += [text]
#      else
        #parts = text.split("\t")
#        row += [text]
#      end
#    end
#    if (row.size>4) and not(row[4]=="W")

#    end

  end

end
