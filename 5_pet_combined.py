import os

contents = []
for file in os.listdir("pet_topical"):
    F = open(os.path.join("pet_topical", file), "r")
    contents.append(F.read())
    F.close()

F = open("pet_united.txt", "w")
F.write("\n\n".join(contents))
F.close()
