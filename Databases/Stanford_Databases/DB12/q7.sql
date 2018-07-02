-- Write an instead-of trigger that enables insertions into view HighlyRated.

create trigger HighlyRatedInsert
instead of insert on HighlyRated
for each row
when exists (SELECT mID, title from Movie
             where Movie.mID = NEW.mID
             and Movie.title = NEW.title)
begin
    insert into Rating values (201, NEW.mID, 5, NULL);
end;