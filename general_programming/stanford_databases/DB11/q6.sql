-- Write a trigger to enforce the following behavior:
-- If A liked B but is updated to A liking C instead, and B and C were friends, make B and C no longer friends.
-- Don't forget to delete the friendship in both directions, and make sure the trigger only runs when the "liked"
-- (ID2) person is changed but the "liking" (ID1) person is not changed.

create trigger grade_update
after update of ID2 on Likes
for each row
when OLD.ID1 = NEW.ID1
begin
    DELETE FROM Friend
    WHERE (Friend.ID1 = OLD.ID2 and Friend.ID2 = NEW.ID2) or
    (Friend.ID2 = OLD.ID2 and Friend.ID1 = NEW.ID2);
 end;
