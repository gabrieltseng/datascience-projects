-- For each student A who likes a student B where the two are not friends, find if they have a friend C in common
-- (who can introduce them!). For all such trios, return the name and grade of A, B, and C.

SELECT distinct A.name, A.grade, B.name, B.grade, C.name, C.grade
FROM Highschooler A, Highschooler B, Highschooler C, Likes
WHERE (A.ID = Likes.ID1) and (B.ID = Likes.ID2)
and (not exists (SELECT 1 FROM Friend WHERE (A.ID = Friend.ID1 and B.ID = Friend.ID2)))
and (exists (SELECT 1 FROM Friend WHERE (C.ID = Friend.ID1 and A.ID = Friend.ID2)))
and (exists (SELECT 1 FROM Friend WHERE (C.ID = Friend.ID1 and B.ID = Friend.ID2)))
