-- Find all students who do not appear in the Likes table (as a student who likes or is liked) and
-- return their names and grades. Sort by grade, then by name within each grade.

SELECT distinct H1.name, H1.grade
FROM Highschooler H1
WHERE H1.ID not in (SELECT H2.ID FROM Highschooler H2, Likes
                     WHERE H2.ID = Likes.ID1)
and H1.ID not in (SELECT H2.ID FROM Highschooler H2, Likes
                     WHERE H2.ID = Likes.ID2)
ORDER BY H1.grade, H1.name
