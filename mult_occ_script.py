f = open("mult_occ_6x6.txt", "r")
f2 = open("mult_occ_12x12.txt", "r")
f3 = open("mult_occ_18x18.txt", "r")
f4 = open("mult_occ_24x24.txt", "r")

line1=f.readline()[:-1]
line2=f2.readline()[:-1]
line3=f3.readline()[:-1]
line4=f4.readline()[:-1]

list1 = [int(x) for x in line1.split(",")]
list2 = [int(x) for x in line2.split(",")]
list3 = [int(x) for x in line3.split(",")]
list4 = [int(x) for x in line4.split(",")]



print(sum(list1)/len(list1))
print(sum(list2)/len(list2))

print(sum(list3)/len(list3))

print(sum(list4)/len(list4))

