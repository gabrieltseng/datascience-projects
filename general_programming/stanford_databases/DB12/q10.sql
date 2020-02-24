-- Write an instead-of trigger that enables deletions from view NoRating.

create trigger NoRatingDelete
instead of delete on NoRating
for each row
begin
    insert into Rating values (201, OLD.mID, 1, NULL);
end;