-- Write a trigger that automatically deletes students when they graduate,
-- i.e., when their grade is updated to exceed 12.

create trigger graduation
after update of grade on Highschooler
for each row
begin
    delete from Highschooler
    where Highschooler.grade > 12;
end;