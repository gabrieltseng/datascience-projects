--  Write an instead-of trigger that enables deletions from view NoRating.

create trigger NoRatingDelete
instead of delete on NoRating
for each row
begin
    delete from Movie
    where Movie.mID = OLD.mID;
end;