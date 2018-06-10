-- For all movies that have an average rating of 4 stars or higher, add 25 to the release year.
-- (Update the existing tuples; don't insert new tuples.)

UPDATE Movie
SET year=year+25
WHERE mID in(
SELECT mID
FROM (SELECT mID, AVG(stars) as star_average FROM Rating Group by mID) as averages
WHERE  star_average >= 4)