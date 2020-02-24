-- Find names and grades of students who only have friends in the same grade. Return the result
-- sorted by grade, then by name within each grade.

SELECT H1.name, H1.grade
FROM Highschooler H1
WHERE not exists (SELECT name FROM Highschooler H2, Friend
                  WHERE Friend.ID1 = H1.ID and Friend.ID2 = H2.ID and H1.grade <> H2.grade)
ORDER BY H1.grade, H1.name