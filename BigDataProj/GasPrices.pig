GASPRICES = load '/user/kovaldk/gas-prices.csv' using PigStorage(',') AS 
		(date: chararray, Adjusted: double, NotAdjusted: double);

GASPRICES = FOREACH GASPRICES GENERATE 
		FLATTEN((tuple(chararray, int, chararray))(STRSPLIT(date, '-')))
		 as (year: chararray, month: int, day: chararray),  NotAdjusted;

GASPRICES = FOREACH GASPRICES GENERATE 
		FLATTEN((tuple(chararray, int))(STRSPLIT(year,'"'))) as (empty: chararray, year),
		month, day, NotAdjusted;

GASPRICES = FOREACH GASPRICES GENERATE 
		year, month, NotAdjusted;

DUMP GASPRICES;
