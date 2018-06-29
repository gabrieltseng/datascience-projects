-- Write one or more triggers to maintain symmetry in friend relationships.
-- Specifically, if (A,B) is deleted from Friend, then (B,A) should be deleted too.
-- If (A,B) is inserted into Friend then (B,A) should be inserted too.

create trigger deletion_symmetry
after delete on Friend
for each row
begin
    delete from Friend
    where ID1 = OLD.ID2 and ID2 = OLD.ID1;
end;

create trigger insertion_symmetry
after insert on Friend
for each row
begin
    insert into Friend values (NEW.ID2, NEW.ID1);
end;