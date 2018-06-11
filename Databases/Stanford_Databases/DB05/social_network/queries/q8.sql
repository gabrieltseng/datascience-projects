-- Find the difference between the number of students in the school and the number of different first names.

SELECT count(distinct ID) - count(distinct name)
FROM Highschooler
