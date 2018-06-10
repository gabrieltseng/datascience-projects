-- Find the difference between the average rating of movies released before 1980 and the average rating of movies
-- released after 1980. (Make sure to calculate the average rating for each movie, then the average of those
-- averages for movies before 1980 and movies after. Don't just calculate the overall average rating before and after 1980.)

SELECT AVG(pre_1980.stars) - AVG(post_1980.stars)
FROM (SELECT AVG(stars) as stars
FROM Rating, Movie
WHERE Movie.year > 1980 and Movie.mID = Rating.mID
GROUP BY Movie.mID) as post_1980,
(SELECT AVG(stars) as stars
FROM Rating, Movie
WHERE Movie.year < 1980 and Movie.mID = Rating.mID
GROUP BY Movie.mID) as pre_1980