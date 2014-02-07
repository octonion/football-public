
begin;

create schema nfl;

create table nfl.trade (
       id    		serial,
       year  		integer,
       trade_date	date,
       primary key (id)
);

create table nfl.team (
       id    	       serial,
       team_name       text,
       primary key (id)
);

create table nfl.trade_pick (
       trade_id         integer,
       team_id		integer,
       pick		integer,
       primary key (trade_id,team_id,pick)
);

commit;
