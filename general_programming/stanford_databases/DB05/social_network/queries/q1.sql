--  Find the names of all students who are friends with someone named Gabriel.

SELECT name
FROM Highschooler H1
WHERE exists (SELECT * FROM Friend, Highschooler H2 where H2.name = 'Gabriel'
            and Friend.ID1 = H1.ID and Friend.ID2 = H2.ID)