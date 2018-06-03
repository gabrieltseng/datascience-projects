-- For every situation where student A likes student B, but we have no information about whom B
-- likes (that is, B does not appear as an ID1 in the Likes table), return A and B's names and grades.

SELECT distinct H1.name, H1.grade, H2.name, H2.grade
FROM Highschooler H1, Highschooler H2, Likes
WHERE Likes.ID1 = H1.ID and Likes.ID2 = H2.ID
and H2.ID not in (SELECT H3.ID FROM Highschooler H3, Likes
                     WHERE H3.ID = Likes.ID1)
