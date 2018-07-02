-- Write an instead-of trigger that enables updates to the stars attribute of view LateRating.

create trigger LateRatingStarsUpdate
instead of update of stars on LateRating
for each row
when OLD.mID = NEW.mID and OLD.ratingDate = NEW.ratingDate
begin
    update Rating
    set stars = NEW.stars
    where Rating.mID = OLD.mID
    and Rating.ratingDate = OLD.ratingDate;
end;