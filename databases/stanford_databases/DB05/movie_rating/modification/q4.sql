-- Remove all ratings where the movie's year is before 1970 or after 2000, and the rating is fewer than 4 stars.

DELETE from Rating
WHERE mID in (SELECT Movie.mID FROM Movie, Rating
              WHERE Movie.year < 1970 or Movie.year > 2000)
      and stars < 4