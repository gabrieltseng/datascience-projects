-- For all cases where the same reviewer rated the same movie twice and gave it a higher rating the second time,
-- return the reviewer's name and the title of the movie.

SELECT name, title
FROM Movie, Reviewer,
    (SELECT R1.mID, R1.rID
    FROM Rating R1, Rating R2
    WHERE (R1.rID = R2.rID) and (R1.ratingDate > R2.ratingDate) and (R1.stars > R2.stars)
          and (R1.mID = R2.mID)) as result
WHERE Movie.mID = result.mID and Reviewer.rID = result.rID