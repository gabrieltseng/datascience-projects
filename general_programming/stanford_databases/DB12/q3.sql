-- Write an instead-of trigger that enables updates to the mID attribute of view LateRating.

create trigger LateRatingmIDUpdate
instead of update of mID on LateRating
for each row
begin
    update Movie
    set mID = NEW.mID
    where Movie.mID = OLD.mID;
    update Rating
    set mID = NEW.mID
    where Rating.mID = OLD.mID;
end;