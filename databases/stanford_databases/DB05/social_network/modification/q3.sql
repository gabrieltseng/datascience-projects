-- For all cases where A is friends with B, and B is friends with C, add a new friendship for the pair A and C.
-- Do not add duplicate friendships, friendships that already exist, or friendships with oneself.

INSERT INTO Friend
SELECT DISTINCT A.ID, C.ID
FROM Highschooler A, Highschooler B, Highschooler C
WHERE exists (SELECT A.ID FROM Friend WHERE Friend.ID1 = A.ID and Friend.ID2 = B.ID)
and exists (SELECT C.ID FROM Friend WHERE Friend.ID1 = C.ID and Friend.ID2 = B.ID)
and A.ID <> C.ID
and not exists (SELECT * FROM Friend WHERE Friend.ID1 = A.ID and Friend.ID2 = C.ID)
