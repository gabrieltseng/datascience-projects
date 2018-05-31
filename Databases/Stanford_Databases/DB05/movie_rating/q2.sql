-- Find all years that have a movie that received a rating of 4 or 5, and sort them in increasing order.

SELECT distinct year
FROM Movie, Rating
WHERE Movie.mID = Rating.mID and stars >= 4
ORDER BY year