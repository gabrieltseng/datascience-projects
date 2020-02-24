-- Write an instead-of trigger that enables updates to the title attribute of view LateRating.

create trigger LateRatingTitleUpdate
instead of update of title on LateRating
for each row
when OLD.mID = NEW.mID
begin
    update Movie
    set title = NEW.title
    where Movie.mID = OLD.mID;
end;