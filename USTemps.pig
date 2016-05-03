GlobalTemps = load 'tempcountry' using PigStorage(',') AS
	 (date: chararray, AverageTemperature: double, Uncertainty: double, Country: chararray);

USGlobalTemps = FILTER GlobalTemps BY Country == 'United States';

newUSGlobalTemps = FOREACH USGlobalTemps GENERATE
			FLATTEN((tuple(int, int, int))STRSPLIT(date, '-')) AS (year:int, month:int, day:int),
			AverageTemperature, Uncertainty, Country;

CurrUSGlobalTemps = FILTER newUSGlobalTemps BY year > 1993;
DUMP CurrUSGlobalTemps;

