-- Write an instead-of trigger that enables deletions from view HighlyRated.

create trigger HighlyRatedDelete
instead of delete on HighlyRated
for each row
begin
    delete from Rating
    where Rating.mID = OLD.mID and Rating.stars > 3;
end;
