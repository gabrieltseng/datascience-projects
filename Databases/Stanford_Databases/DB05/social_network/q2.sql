-- For every student who likes someone 2 or more grades younger than themselves,
-- return that student's name and grade, and the name and grade of the student they like.

SELECT H1.name, H1.grade, H2.name, H2.grade
FROM Highschooler H1, Highschooler H2, Likes
Where Likes.ID1 = H1.ID and Likes.ID2 = H2.ID and H1.grade >= (H2.grade + 2)