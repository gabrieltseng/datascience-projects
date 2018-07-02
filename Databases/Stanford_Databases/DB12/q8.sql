-- Write an instead-of trigger that enables insertions into view NoRating.

create trigger NoRatingInsert
instead of insert on NoRating
for each row
when exists (SELECT mID, title from Movie
             where Movie.mID = NEW.mID
             and Movie.title = NEW.title)
begin
    delete from Rating
    where Rating.mID = NEW.mID;
end;