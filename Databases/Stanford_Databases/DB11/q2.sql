-- Write one or more triggers to manage the grade attribute of new Highschoolers.
-- If the inserted tuple has a value less than 9 or greater than 12, change the value to NULL.
-- On the other hand, if the inserted tuple has a null value for grade, change it to 9.

create trigger null_grade
after insert on Highschooler
for each row
when NEW.grade is NULL
begin
    update Highschooler
    set grade = 9
    where ID = NEW.ID;
end;

create trigger wrong_grade
after insert on Highschooler
for each row
when NEW.grade < 9 or NEW.grade > 12
begin
    update Highschooler
    set grade = NULL
    where ID = NEW.ID;
end;