f = open("mult_occ_6x6.txt", "r")
line=f.readline()
f.close()
list = [int(x) for x in line.split(",")]
print(sum(list)/len(list))