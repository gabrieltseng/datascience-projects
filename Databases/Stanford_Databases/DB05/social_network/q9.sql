-- Find the name and grade of all students who are liked by more than one other student.

SELECT name, grade
FROM Highschooler H1
WHERE (SELECT count(H2.ID) FROM Highschooler H2, Likes WHERE H2.ID = Likes.ID1 and H1.ID = Likes.ID2) > 1