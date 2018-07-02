-- Write an instead-of trigger that enables deletions from view HighlyRated.

create trigger HighlyRatedDelete
instead of delete on HighlyRated
for each row
begin
    update Rating
    set stars = 3
    where Rating.mID = OLD.mID and Rating.stars > 3;
end;