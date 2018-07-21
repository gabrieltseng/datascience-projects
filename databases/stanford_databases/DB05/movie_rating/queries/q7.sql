-- For each movie that has at least one rating, find the highest number of stars that movie received.
-- Return the movie title and number of stars. Sort by movie title.

SELECT distinct title, stars
FROM Rating R1, Movie
WHERE (Movie.mID = R1.mID) and not exists (SELECT * FROM Rating R2 WHERE R2.mID = R1.mID
and R2.stars > R1.stars)
ORDER BY title