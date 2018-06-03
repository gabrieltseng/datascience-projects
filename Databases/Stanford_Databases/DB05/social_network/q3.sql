-- For every pair of students who both like each other, return the name and grade of both students.
-- Include each pair only once, with the two names in alphabetical order.



SELECT H2.name, H2.grade, H1.name, H1.grade
FROM Highschooler H1, Likes, Highschooler H2
WHERE Likes.ID1 = H1.ID and Likes.ID2 = H2.ID and H1.name > H2.name
INTERSECT
SELECT H1.name, H1.grade, H2.name, H2.grade
FROM Highschooler H1, Likes, Highschooler H2
WHERE Likes.ID1 = H1.ID and Likes.ID2 = H2.ID and H2.name > H1.name
