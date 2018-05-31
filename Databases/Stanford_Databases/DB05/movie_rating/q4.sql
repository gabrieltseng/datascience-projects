-- Some reviewers didn't provide a date with their rating. Find the names of all reviewers who have ratings with a NULL value for the date.

SELECT distinct name
FROM Reviewer, Rating
WHERE Reviewer.rID = Rating.rID and ratingDate is null