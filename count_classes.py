import os

count0 = 0
count1 = 0

for file in os.listdir("dataset/annotations"):
    if file.endswith(".txt"):
        with open(os.path.join("dataset/annotations", file), "r") as f:
            for line in f:
                cls = int(line.split()[0])

                if cls == 0:
                    count0 += 1
                elif cls == 1:
                    count1 += 1

print("Supporting towers (class 0):", count0)
print("Monopole towers (class 1):", count1)