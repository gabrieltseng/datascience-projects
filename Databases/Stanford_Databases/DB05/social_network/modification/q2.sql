-- If two students A and B are friends, and A likes B but not vice-versa, remove the Likes tuple.

DELETE from Likes
WHERE exists (SELECT ID1, ID2 from Friend where Likes.ID1 = Friend.ID1 and Likes.ID2 = Friend.ID2)
and not exists (SELECT ID1, ID2 FROM Likes L2 where L2.ID1 = Likes.ID2 and L2.ID2 = Likes.ID1)