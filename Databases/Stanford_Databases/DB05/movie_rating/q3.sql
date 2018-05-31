-- Find the titles of all movies that have no ratings.

SELECT title
FROM Movie M1
WHERE not exists (SELECT * FROM Movie M2, Rating WHERE M1.mID = Rating.mID
                                                   and M1.title = M2.title)