-- Write a trigger that makes new students named 'Friendly' automatically like everyone else in their grade.

create trigger make_friendly
after insert on Highschooler
for each row
when NEW.name = 'Friendly'
begin
    insert into Likes SELECT NEW.ID, other.ID FROM Highschooler as other
    WHERE other.grade = NEW.grade and other.name <> NEW.name;
end;