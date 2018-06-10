--  Insert 5-star ratings by James Cameron for all movies in the database. Leave the review date as NULL.

INSERT into Rating
SELECT 207, mID, 5, null
FROM Movie
