-- Write a trigger that automatically deletes students when they graduate, i.e., when their grade is updated to
-- exceed 12 (same as Question 4).
-- In addition, write a trigger so when a student is moved ahead one grade, then so are all of his or her friends.

create trigger graduation
after update of grade on Highschooler
for each row
begin
    delete from Highschooler
    where Highschooler.grade > 12;
end;

create trigger grade_update
after update of grade on Highschooler
for each row
begin
    update Highschooler
    set grade = grade + 1
    where ID in (SELECT ID1 from Friend where Friend.ID2 = OLD.ID);
 end;