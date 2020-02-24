-- Finally, write a single instead-of trigger that combines all three of the previous triggers to enable simultaneous
-- updates to attributes mID, title, and/or stars in view LateRating.

create trigger LateRatingUpdate
instead of update on LateRating
for each row
when OLD.ratingDate = NEW.ratingDate
begin
    update Movie
    set mID = NEW.mID, title = NEW.title
    where Movie.mID = OLD.mID;
    update Rating
    set stars = NEW.stars
    where Rating.mID = OLD.mID
    and Rating.ratingDate = OLD.ratingDate;
    update Rating
    set mID = NEW.mID
    where Rating.mID = OLD.mID;
end;
