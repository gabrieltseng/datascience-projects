## DB12: Views and Authorization

The statement of accomplishment can be viewed [here](DB12_Statement.pdf)

The exercises consisted of implementing the triggers (described in words at the top of the file), using `SQLite`,
on the [movie rating](rating.sql) data, so that the following views could be modified:

View **LateRating** contains movie ratings after January 20, 2011. The view contains the movie ID, movie title, number 
of stars, and rating date.

```sql
create view LateRating as
  select distinct R.mID, title, stars, ratingDate
  from Rating R, Movie M
  where R.mID = M.mID
  and ratingDate > '2011-01-20'
```

View **HighlyRated** contains movies with at least one rating above 3 stars. The view contains the movie ID and movie title.
```sql
create view HighlyRated as
  select mID, title
  from Movie
  where mID in (select mID from Rating where stars > 3)
```

View **NoRating** contains movies with no ratings in the database. The view contains the movie ID and movie title.

```sql
create view NoRating as
  select mID, title
  from Movie
  where mID not in (select mID from Rating) 
```