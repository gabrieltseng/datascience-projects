-- For each movie, return the title and the 'rating spread', that is, the difference between highest and lowest
-- ratings given to that movie. Sort by rating spread from highest to lowest, then by movie title.

SELECT distinct max_rating.title, max_rating.stars - min_rating.stars as spread
FROM (SELECT distinct title, stars
FROM Rating R1, Movie
WHERE (Movie.mID = R1.mID) and not exists (SELECT * FROM Rating R2 WHERE R2.mID = R1.mID
and R2.stars > R1.stars)) as max_rating,
(SELECT distinct title, stars
FROM Rating R1, Movie
WHERE (Movie.mID = R1.mID) and not exists (SELECT * FROM Rating R2 WHERE R2.mID = R1.mID
and R2.stars < R1.stars)) as min_rating
WHERE min_rating.title = max_rating.title
ORDER BY spread DESC, max_rating.title